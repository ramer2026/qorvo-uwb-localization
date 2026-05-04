# analyze_qorvo_complete.py
#
# Step 2 of the pipeline: full analysis + machine learning models on the r1 dataset.
#
# Inputs:  raw/  folder (re-parses the data itself, similar to process_qorvo_dataset.py)
# Outputs: processed/  CSV files including measurements_clean.csv (used by all later scripts)
#          plots/  PNG plots comparing LOS vs obstructed signal distributions
#
# This script does everything in one go:
#   1. Scan and parse all raw files
#   2. Compute LOS vs obstruction statistics
#   3. Train Random Forest localization models (predict 3D position from signals)
#   4. Train LOS/obstruction classifier
#   5. Generate comparison plots
#
# Run with: python scripts/analyze_qorvo_complete.py

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report


RAW_DIR = Path("raw")
PROCESSED_DIR = Path("processed")
PLOTS_DIR = Path("plots")

PROCESSED_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Matches the filename convention: {point}_{condition}_{anchor}_{kind}
FILENAME_RE = re.compile(
    r"^(?P<point>.+)_(?P<condition>los|obst)_(?P<anchor>anchorA|anchorB)_(?P<kind>stdout\.txt|diag\.json)$"
)

# Each point/condition must have these 4 files to be considered complete
REQUIRED = {
    ("anchorA", "stdout.txt"),
    ("anchorA", "diag.json"),
    ("anchorB", "stdout.txt"),
    ("anchorB", "diag.json"),
}


def parse_coord_token(tok: str, axis: str) -> float:
    """
    Convert a coordinate token like 'xm050' to -0.50 meters.
    'x050' -> +0.50, 'xm050' -> -0.50
    """
    assert tok.startswith(axis), tok
    rest = tok[len(axis):]
    sign = 1.0
    if rest.startswith("m"):
        sign = -1.0
        rest = rest[1:]
    return sign * (int(rest) / 100.0)


def parse_point(point: str) -> dict[str, float]:
    """Parse a point string like 'z050_xm050_y100' into {'z_m': 0.5, 'x_m': -0.5, 'y_m': 1.0}."""
    parts = point.split("_")
    ztok = next(p for p in parts if p.startswith("z"))
    xtok = next(p for p in parts if p.startswith("x"))
    ytok = next(p for p in parts if p.startswith("y"))
    return {
        "z_m": parse_coord_token(ztok, "z"),
        "x_m": parse_coord_token(xtok, "x"),
        "y_m": parse_coord_token(ytok, "y"),
    }


def scan_files() -> pd.DataFrame:
    """
    Scan the raw/ directory and classify each file as recognized or not.
    Returns a DataFrame with one row per file including parsed metadata.
    """
    rows = []
    for p in RAW_DIR.iterdir():
        if not p.is_file():
            continue
        m = FILENAME_RE.match(p.name)
        if not m:
            rows.append({
                "filename": p.name,
                "point": None,
                "condition": None,
                "anchor": None,
                "kind": None,
                "recognized": False,
            })
            continue

        d = m.groupdict()
        rows.append({
            "filename": p.name,
            "path": str(p),
            "point": d["point"],
            "condition": d["condition"],
            "anchor": d["anchor"],
            "kind": d["kind"],
            "recognized": True,
        })

    return pd.DataFrame(rows)


def completeness_report(files_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check which (point, condition) combos have all 4 required files.
    Missing files are flagged so the user knows which data is incomplete.
    """
    known = files_df[files_df["recognized"]].copy()
    points = sorted(known["point"].dropna().unique())
    rows = []

    for point in points:
        for condition in ["los", "obst"]:
            subset = known[(known["point"] == point) & (known["condition"] == condition)]
            have = set(zip(subset["anchor"], subset["kind"]))
            missing = sorted(REQUIRED - have)
            coords = parse_point(point)
            rows.append({
                "point": point,
                **coords,
                "condition": condition,
                "complete": len(missing) == 0,
                "n_files": len(subset),
                "missing": ";".join([f"{a}_{k}" for a, k in missing]),
            })

    return pd.DataFrame(rows)


def parse_stdout_file(path: Path, point: str, condition: str, anchor: str) -> pd.DataFrame:
    """
    Parse a ranging stdout .txt file into a DataFrame of individual measurements.
    Each measurement block starts with 'sequence n:' and contains distance,
    angle-of-arrival (AoA), RSSI (signal strength), and status fields.
    """
    text = path.read_text(errors="replace").splitlines()
    rows = []
    current: dict[str, Any] | None = None

    for line in text:
        s = line.strip()

        if s.startswith("sequence n:"):
            if current is not None and "sequence_n" in current:
                rows.append(current)
            current = {
                "point": point,
                "condition": condition,
                "anchor": anchor,
                **parse_point(point),
            }
            try:
                current["sequence_n"] = int(s.split(":")[-1].strip())
            except Exception:
                pass

        if current is None:
            continue

        if s.startswith("status:"):
            current["status"] = s.split(":", 1)[1].strip()
            current["ok"] = "Ok (0x0)" in current["status"]

        elif s.startswith("distance:"):
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*cm", s)
            if m:
                current["distance_cm"] = float(m.group(1))

        elif s.startswith("AoA azimuth:"):
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*deg", s)
            if m:
                current["aoa_azimuth_deg"] = float(m.group(1))

        elif s.startswith("AoA az. FOM:"):
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*%", s)
            if m:
                current["aoa_az_fom_pct"] = float(m.group(1))

        elif s.startswith("AoA elevation:"):
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*deg", s)
            if m:
                current["aoa_elevation_deg"] = float(m.group(1))

        elif s.startswith("AoA elev. FOM:"):
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*%", s)
            if m:
                current["aoa_elev_fom_pct"] = float(m.group(1))

        elif s.startswith("AoA dest azimuth:"):
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*deg", s)
            if m:
                current["aoa_dest_azimuth_deg"] = float(m.group(1))

        elif s.startswith("AoA dest elevation:"):
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*deg", s)
            if m:
                current["aoa_dest_elevation_deg"] = float(m.group(1))

        elif s.startswith("rssi:"):
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*dBm", s)
            if m:
                current["rssi_dbm"] = float(m.group(1))

    if current is not None and "sequence_n" in current:
        rows.append(current)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep raw rows; clean filtering happens later.
    return df


def flatten_diag_sequence(obj: dict[str, Any], point: str, condition: str, anchor: str) -> dict[str, Any]:
    """
    Flatten one ranging sequence from a diagnostic JSON into a single summary row.

    The JSON has nested reports > fields > segment_metrics / aoa lists.
    We average all values within each sequence to get one number per metric per sequence.
    AoA type 0/1/2 correspond to x/y/z axes respectively.
    """
    row: dict[str, Any] = {
        "point": point,
        "condition": condition,
        "anchor": anchor,
        **parse_point(point),
        "sequence_n": obj.get("sequence_n"),
    }

    rsl_vals = []
    path1_rsl_vals = []
    path1_snr_vals = []
    peak_snr_vals = []
    peak_rsl_vals = []
    cfo_vals = []

    # aoa_type 0=x, 1=y, 2=z axis angles
    aoa_type_values: dict[int, list[float]] = {}
    aoa_type_foms: dict[int, list[float]] = {}

    for report in obj.get("reports", []):
        for field in report.get("fields", []):
            if "cfo" in field:
                cfo_vals.append(field["cfo"])

            if "segment_metrics" in field:
                for sm in field["segment_metrics"]:
                    if "rsl_dbm" in sm:
                        rsl_vals.append(sm["rsl_dbm"])
                    if "path1_rsl_dbm" in sm:
                        path1_rsl_vals.append(sm["path1_rsl_dbm"])
                    if "path1_snr" in sm:
                        path1_snr_vals.append(sm["path1_snr"])
                    if "peak_snr" in sm:
                        peak_snr_vals.append(sm["peak_snr"])
                    if "peak_rsl_dbm" in sm:
                        peak_rsl_vals.append(sm["peak_rsl_dbm"])

            if "aoa" in field:
                for a in field["aoa"]:
                    t = int(a.get("aoa_type", -1))
                    if "aoa" in a:
                        aoa_type_values.setdefault(t, []).append(a["aoa"])
                    if "aoa_fom" in a:
                        aoa_type_foms.setdefault(t, []).append(a["aoa_fom"])

    def mean_or_nan(vals):
        return float(np.mean(vals)) if vals else np.nan

    def std_or_nan(vals):
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan

    # Store averaged diagnostic features as diag_* columns
    row["diag_cfo_mean"] = mean_or_nan(cfo_vals)
    row["diag_rsl_dbm_mean"] = mean_or_nan(rsl_vals)
    row["diag_path1_rsl_dbm_mean"] = mean_or_nan(path1_rsl_vals)
    row["diag_path1_snr_mean"] = mean_or_nan(path1_snr_vals)
    row["diag_peak_rsl_dbm_mean"] = mean_or_nan(peak_rsl_vals)
    row["diag_peak_snr_mean"] = mean_or_nan(peak_snr_vals)

    for t in [0, 1, 2]:
        row[f"diag_aoa_type{t}_deg_mean"] = mean_or_nan(aoa_type_values.get(t, []))
        row[f"diag_aoa_type{t}_deg_std"] = std_or_nan(aoa_type_values.get(t, []))
        row[f"diag_aoa_type{t}_fom_mean"] = mean_or_nan(aoa_type_foms.get(t, []))

    return row


def parse_diag_file(path: Path, point: str, condition: str, anchor: str) -> pd.DataFrame:
    """Parse a diagnostic JSON file; return one row per ranging sequence."""
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception as e:
        print(f"WARNING: failed to parse JSON {path}: {e}")
        return pd.DataFrame()

    if not isinstance(data, list):
        return pd.DataFrame()

    rows = [flatten_diag_sequence(obj, point, condition, anchor) for obj in data if isinstance(obj, dict)]
    return pd.DataFrame(rows)


def build_measurements(files_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse all recognized files and merge stdout + diagnostic data.
    Returns (stdout_df, diag_df, merged_df).
    The merged DataFrame is the main input to all downstream analysis.
    """
    stdout_frames = []
    diag_frames = []

    for _, r in files_df[files_df["recognized"]].iterrows():
        path = Path(r["path"])
        point, condition, anchor, kind = r["point"], r["condition"], r["anchor"], r["kind"]

        if kind == "stdout.txt":
            df = parse_stdout_file(path, point, condition, anchor)
            if not df.empty:
                stdout_frames.append(df)

        elif kind == "diag.json":
            df = parse_diag_file(path, point, condition, anchor)
            if not df.empty:
                diag_frames.append(df)

    stdout_df = pd.concat(stdout_frames, ignore_index=True) if stdout_frames else pd.DataFrame()
    diag_df = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()

    # Left join on sequence_n so every ranging row gets its diagnostic features
    merged = stdout_df.merge(
        diag_df,
        on=["point", "condition", "anchor", "x_m", "y_m", "z_m", "sequence_n"],
        how="left",
        suffixes=("", "_diag"),
    )

    return stdout_df, diag_df, merged


def clean_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid measurements:
      - status must be 'Ok (0x0)'
      - distance must be positive (zero indicates a failed ranging)
      - RSSI must not be exactly 0.0 (indicates a missing reading)
    """
    out = df.copy()
    if "ok" in out:
        out = out[out["ok"] == True]
    if "distance_cm" in out:
        out = out[out["distance_cm"].fillna(0) > 0]
    if "rssi_dbm" in out:
        out = out[out["rssi_dbm"].fillna(0) != -0.0]
    return out.reset_index(drop=True)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d effect size: (mean_b - mean_a) / pooled_std.
    Quantifies how different LOS vs obstructed distributions are,
    independent of sample size (unlike p-values).
    """
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (np.mean(b) - np.mean(a)) / pooled


def los_vs_obst_stats(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Statistical comparison of every signal feature between LOS and obstructed conditions,
    for every (point, anchor) pair.

    Uses both Welch's t-test and Mann-Whitney U test (non-parametric alternative)
    to check significance, plus Cohen's d for effect size.
    """
    features = [
        "distance_cm",
        "rssi_dbm",
        "aoa_azimuth_deg",
        "aoa_elevation_deg",
        "aoa_dest_azimuth_deg",
        "aoa_dest_elevation_deg",
        "diag_rsl_dbm_mean",
        "diag_path1_rsl_dbm_mean",
        "diag_path1_snr_mean",
        "diag_peak_snr_mean",
        "diag_aoa_type0_deg_mean",
        "diag_aoa_type1_deg_mean",
    ]

    rows = []
    for point in sorted(clean["point"].unique()):
        for anchor in sorted(clean["anchor"].unique()):
            los = clean[(clean["point"] == point) & (clean["condition"] == "los") & (clean["anchor"] == anchor)]
            obst = clean[(clean["point"] == point) & (clean["condition"] == "obst") & (clean["anchor"] == anchor)]
            if los.empty or obst.empty:
                continue

            for feat in features:
                if feat not in clean.columns:
                    continue
                a = los[feat].dropna().to_numpy(float)
                b = obst[feat].dropna().to_numpy(float)
                if len(a) < 5 or len(b) < 5:
                    continue

                try:
                    t_p = float(ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)
                except Exception:
                    t_p = np.nan

                try:
                    mw_p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
                except Exception:
                    mw_p = np.nan

                rows.append({
                    "point": point,
                    **parse_point(point),
                    "anchor": anchor,
                    "feature": feat,
                    "n_los": len(a),
                    "n_obst": len(b),
                    "los_mean": float(np.mean(a)),
                    "los_std": float(np.std(a, ddof=1)),
                    "obst_mean": float(np.mean(b)),
                    "obst_std": float(np.std(b, ddof=1)),
                    "delta_obst_minus_los": float(np.mean(b) - np.mean(a)),
                    "cohen_d_obst_minus_los": cohen_d(a, b),
                    "welch_p": t_p,
                    "mannwhitney_p": mw_p,
                    "significant_p05": bool(t_p < 0.05) if np.isfinite(t_p) else False,
                })

    return pd.DataFrame(rows)


def point_summary(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and std of each feature per (point, condition, anchor, coordinates).
    This is the aggregated table that gets reshaped into ML model inputs.
    """
    agg_features = [
        "distance_cm",
        "rssi_dbm",
        "aoa_azimuth_deg",
        "aoa_elevation_deg",
        "diag_rsl_dbm_mean",
        "diag_path1_snr_mean",
        "diag_peak_snr_mean",
        "diag_aoa_type0_deg_mean",
        "diag_aoa_type1_deg_mean",
    ]
    present = [c for c in agg_features if c in clean.columns]

    grouped = clean.groupby(["point", "condition", "anchor", "x_m", "y_m", "z_m"])
    rows = []
    for keys, g in grouped:
        point, condition, anchor, x, y, z = keys
        row = {
            "point": point,
            "condition": condition,
            "anchor": anchor,
            "x_m": x,
            "y_m": y,
            "z_m": z,
            "n": len(g),
        }
        for c in present:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std()
        rows.append(row)

    return pd.DataFrame(rows)


def plot_feature_by_point(stats: pd.DataFrame, feature: str):
    """
    Bar chart showing the LOS-to-obstructed mean shift for each (point, anchor) combination.
    Bars above zero mean obstruction increases the metric; below zero means it decreases it.
    Saved to plots/delta_{feature}.png.
    """
    sub = stats[stats["feature"] == feature].copy()
    if sub.empty:
        return

    sub["label"] = sub["point"] + "_" + sub["anchor"]
    sub = sub.sort_values(["z_m", "x_m", "y_m", "anchor"])

    plt.figure(figsize=(max(10, len(sub) * 0.25), 5))
    plt.axhline(0, linewidth=1)
    plt.bar(np.arange(len(sub)), sub["delta_obst_minus_los"])
    plt.xticks(np.arange(len(sub)), sub["label"], rotation=90, fontsize=7)
    plt.ylabel(f"Obst - LOS delta: {feature}")
    plt.title(f"LOS vs obstruction shift: {feature}")
    plt.tight_layout()
    out = PLOTS_DIR / f"delta_{feature}.png"
    plt.savefig(out, dpi=180)
    plt.close()


def plot_condition_box(clean: pd.DataFrame, feature: str):
    """
    Box plot comparing LOS vs obstructed distributions for a single feature.
    Outliers hidden (showfliers=False) to keep the plot readable.
    Saved to plots/box_{feature}.png.
    """
    if feature not in clean.columns:
        return

    data_los = clean[clean["condition"] == "los"][feature].dropna()
    data_obst = clean[clean["condition"] == "obst"][feature].dropna()
    if data_los.empty or data_obst.empty:
        return

    plt.figure(figsize=(6, 5))
    plt.boxplot([data_los, data_obst], labels=["LOS", "Obstructed"], showfliers=False)
    plt.ylabel(feature)
    plt.title(f"{feature}: LOS vs obstructed")
    plt.tight_layout()
    out = PLOTS_DIR / f"box_{feature}.png"
    plt.savefig(out, dpi=180)
    plt.close()


def build_model_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the per-anchor summary into one row per (point, condition) for ML input.
    Each row represents one measurement location under one condition, with features
    from anchorA and anchorB as separate columns (e.g., anchorA_distance_cm_mean).

    Only keeps rows where both anchors are present — partial data would skew the model.
    """
    base_cols = ["point", "condition", "x_m", "y_m", "z_m"]
    feature_cols = [c for c in summary.columns if c.endswith("_mean") and c not in base_cols]

    rows = []
    for (point, condition), g in summary.groupby(["point", "condition"]):
        if set(g["anchor"]) < {"anchorA", "anchorB"}:
            continue

        first = g.iloc[0]
        row = {
            "point": point,
            "condition": condition,
            "is_obst": 1 if condition == "obst" else 0,
            "x_m": first["x_m"],
            "y_m": first["y_m"],
            "z_m": first["z_m"],
        }

        for _, r in g.iterrows():
            a = r["anchor"]
            for c in feature_cols:
                row[f"{a}_{c}"] = r[c]

        rows.append(row)

    return pd.DataFrame(rows)


def run_localization_models(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train and evaluate Random Forest regression models for 3D localization.
    Tests three feature sets of increasing richness:
      1. range_printed_aoa: distance + printed AoA only (minimal)
      2. range_aoa_rssi_diag_no_condition: adds RSSI and diagnostic signal quality
      3. range_aoa_rssi_diag_condition: also includes whether signal is obstructed

    Uses GroupKFold cross-validation grouped by point — this ensures the model is
    evaluated on points it has never seen during training, simulating real deployment.
    Without this grouping, the model would appear unrealistically accurate because it
    memorizes individual measurement noise at each point.

    Returns a DataFrame of error metrics per feature set.
    """
    target_cols = ["x_m", "y_m", "z_m"]
    rows = []

    if len(model_df) < 10:
        return pd.DataFrame()

    groups = model_df["point"].astype(str).to_numpy()
    y = model_df[target_cols].to_numpy(float)

    all_features = [
        c for c in model_df.columns
        if c not in ["point", "condition", "x_m", "y_m", "z_m"]
        and pd.api.types.is_numeric_dtype(model_df[c])
    ]

    # Feature set 1: only what the device prints — distance + horizontal/vertical angle
    simple_features = [
        c for c in all_features
        if any(k in c for k in ["distance_cm", "aoa_azimuth", "aoa_elevation"])
    ]

    # Feature set 2: adds RSSI and signal quality diagnostics (no condition label)
    diagnostic_features_no_condition = [
        c for c in all_features
        if any(k in c for k in ["distance_cm", "aoa", "rssi", "rsl", "snr"])
        and c != "is_obst"
    ]

    # Feature set 3: also gives the model the LOS/obstructed label as a feature
    diagnostic_features_with_condition = [
        c for c in diagnostic_features_no_condition + ["is_obst"]
        if c in all_features
    ]

    feature_sets = {
        "range_printed_aoa": simple_features,
        "range_aoa_rssi_diag_no_condition": diagnostic_features_no_condition,
        "range_aoa_rssi_diag_condition": diagnostic_features_with_condition,
    }

    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    for name, feats in feature_sets.items():
        feats = [f for f in feats if f in model_df.columns]
        if not feats:
            continue

        X = model_df[feats].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))  # Fill NaNs with column medians

        # Run cross-validation: collect predictions for all test folds
        preds = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in cv.split(X, y, groups=groups):
            model = RandomForestRegressor(n_estimators=300, random_state=1, min_samples_leaf=2)
            model.fit(X.iloc[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X.iloc[test_idx])

        # 3D error = Euclidean distance between predicted and true XYZ position
        err = np.linalg.norm(preds - y, axis=1)
        rows.append({
            "model": name,
            "n_samples": len(X),
            "n_features": len(feats),
            "mean_3d_error_m": float(np.mean(err)),
            "median_3d_error_m": float(np.median(err)),
            "rmse_x_m": float(math.sqrt(mean_squared_error(y[:, 0], preds[:, 0]))),
            "rmse_y_m": float(math.sqrt(mean_squared_error(y[:, 1], preds[:, 1]))),
            "rmse_z_m": float(math.sqrt(mean_squared_error(y[:, 2], preds[:, 2]))),
        })

        pred_df = model_df[["point", "condition", "x_m", "y_m", "z_m"]].copy()
        pred_df[["pred_x_m", "pred_y_m", "pred_z_m"]] = preds
        pred_df["error_3d_m"] = err
        pred_df.to_csv(PROCESSED_DIR / f"localization_predictions_{name}.csv", index=False)

    return pd.DataFrame(rows)


def run_obstruction_classifier(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a Random Forest classifier to predict whether a measurement is LOS or obstructed.
    Uses leave-one-point-out cross-validation (GroupKFold by point) for honest evaluation.

    This classifier could be used in a real system to flag when a signal path is blocked
    before trusting its distance/angle measurements for localization.
    """
    if len(model_df) < 10:
        return pd.DataFrame()

    y = model_df["is_obst"].to_numpy(int)
    groups = model_df["point"].astype(str).to_numpy()

    feats = [
        c for c in model_df.columns
        if c not in ["point", "condition", "is_obst", "x_m", "y_m", "z_m"]
        and pd.api.types.is_numeric_dtype(model_df[c])
    ]

    X = model_df[feats].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    preds = np.zeros_like(y)
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        clf = RandomForestClassifier(n_estimators=300, random_state=2, min_samples_leaf=2)
        clf.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X.iloc[test_idx])

    acc = accuracy_score(y, preds)

    out = pd.DataFrame({
        "point": model_df["point"],
        "condition": model_df["condition"],
        "true_is_obst": y,
        "pred_is_obst": preds,
        "correct": y == preds,
    })
    out.to_csv(PROCESSED_DIR / "obstruction_classifier_predictions.csv", index=False)

    return pd.DataFrame([{
        "classifier": "random_forest_grouped_by_point",
        "n_samples": len(X),
        "n_features": len(feats),
        "accuracy": float(acc),
    }])


def main():
    print("Scanning files...")
    files_df = scan_files()
    files_df.to_csv(PROCESSED_DIR / "file_scan.csv", index=False)

    complete = completeness_report(files_df)
    complete.to_csv(PROCESSED_DIR / "file_completeness.csv", index=False)

    print("\nCompleteness:")
    print(complete.pivot(index="point", columns="condition", values="complete").fillna(False))

    print("\nParsing stdout + diag JSON...")
    stdout_df, diag_df, merged = build_measurements(files_df)
    stdout_df.to_csv(PROCESSED_DIR / "stdout_measurements_raw.csv", index=False)
    diag_df.to_csv(PROCESSED_DIR / "diag_measurements_raw.csv", index=False)
    merged.to_csv(PROCESSED_DIR / "measurements_merged_raw.csv", index=False)

    # This is the primary cleaned dataset used by all downstream scripts
    clean = clean_measurements(merged)
    clean.to_csv(PROCESSED_DIR / "measurements_clean.csv", index=False)

    print(f"\nRaw merged rows: {len(merged)}")
    print(f"Clean usable rows: {len(clean)}")
    print(clean.groupby(["condition", "anchor"]).size())

    print("\nComputing LOS vs obstruction stats...")
    stats = los_vs_obst_stats(clean)
    stats.to_csv(PROCESSED_DIR / "los_vs_obst_stats.csv", index=False)

    summary = point_summary(clean)
    summary.to_csv(PROCESSED_DIR / "point_summary.csv", index=False)

    # Count how often each feature shows a significant LOS vs obstructed difference
    sig_summary = (
        stats.groupby("feature")["significant_p05"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "n_significant", "count": "n_comparisons"})
    )
    sig_summary["fraction_significant"] = sig_summary["n_significant"] / sig_summary["n_comparisons"]
    sig_summary.to_csv(PROCESSED_DIR / "significance_summary.csv", index=False)

    print("\nSignificance summary:")
    print(sig_summary.sort_values("fraction_significant", ascending=False).to_string(index=False))

    print("\nGenerating plots...")
    for feat in [
        "distance_cm",
        "rssi_dbm",
        "diag_rsl_dbm_mean",
        "diag_path1_snr_mean",
        "diag_peak_snr_mean",
        "aoa_azimuth_deg",
        "aoa_elevation_deg",
        "diag_aoa_type0_deg_mean",
        "diag_aoa_type1_deg_mean",
    ]:
        plot_feature_by_point(stats, feat)
        plot_condition_box(clean, feat)

    print("\nBuilding model table...")
    model_df = build_model_table(summary)
    model_df.to_csv(PROCESSED_DIR / "model_table_point_condition.csv", index=False)

    print("\nRunning localization models...")
    loc_results = run_localization_models(model_df)
    loc_results.to_csv(PROCESSED_DIR / "localization_model_results.csv", index=False)
    if not loc_results.empty:
        print(loc_results.to_string(index=False))

    print("\nRunning LOS/obstruction classifier...")
    clf_results = run_obstruction_classifier(model_df)
    clf_results.to_csv(PROCESSED_DIR / "obstruction_classifier_results.csv", index=False)
    if not clf_results.empty:
        print(clf_results.to_string(index=False))

    print("\nDONE.")
    print("Key outputs:")
    print("  processed/measurements_clean.csv")
    print("  processed/point_summary.csv")
    print("  processed/los_vs_obst_stats.csv")
    print("  processed/significance_summary.csv")
    print("  processed/localization_model_results.csv")
    print("  processed/obstruction_classifier_results.csv")
    print("  plots/*.png")


if __name__ == "__main__":
    main()
