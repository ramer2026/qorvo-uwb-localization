# trial_generalization_analysis.py
#
# Step 6 of the pipeline: cross-trial generalization test.
#
# Inputs:  trials/r1/raw/  — first collection session (24 points, full dataset)
#          trials/r2/raw/  — second collection session (4 points, repeated placement)
# Outputs: processed/trials_r1_r2_measurements_clean.csv        — combined r1+r2 data
#          processed/trial_generalization_r1_to_r2_results.csv  — aggregate error metrics
#          processed/trial_generalization_r1_to_r2_predictions.csv — per-row predictions
#          processed/trial_generalization_r1_to_r2_per_point.csv   — per-point summary
#
# Why this matters:
#   The previous scripts train and test on the same dataset (using cross-validation).
#   This script tests a harder and more realistic scenario: the model is trained on
#   measurements from one physical collection session (r1) and tested on an entirely
#   separate collection session (r2) at the same locations.
#
#   Real-world deployment involves this kind of shift: you collect training data once,
#   then the system must work when deployed later. The r2 data was collected independently
#   (different day, slightly different device placement, etc.) at 4 of the 24 r1 points.
#
# Run with: python scripts/trial_generalization_analysis.py

from __future__ import annotations
import json
import math
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score

OUT = Path("processed")
OUT.mkdir(exist_ok=True)

FILENAME_RE = re.compile(
    r"^(?P<point>.+)_(?P<condition>los|obst)_(?P<anchor>anchorA|anchorB)_(?P<kind>stdout\.txt|diag\.json)$"
)

# Subset of features available in both stdout and diagnostic files
FEATURES = [
    "distance_cm",               # Ranging distance measurement
    "rssi_dbm",                  # Signal strength
    "aoa_azimuth_deg",           # Horizontal angle of arrival
    "aoa_elevation_deg",         # Vertical angle of arrival
    "diag_rsl_dbm_mean",         # Average received signal level
    "diag_path1_rsl_dbm_mean",   # Direct-path signal level
    "diag_path1_snr_mean",       # Direct-path SNR
    "diag_peak_snr_mean",        # Peak SNR
    "diag_aoa_type0_deg_mean",   # Raw AoA x-axis
    "diag_aoa_type1_deg_mean",   # Raw AoA y-axis
]


def parse_coord_token(tok, axis):
    """Convert a coordinate token like 'xm050' to -0.50 meters."""
    rest = tok[len(axis):]
    sign = 1.0
    if rest.startswith("m"):
        sign = -1.0
        rest = rest[1:]
    return sign * int(rest) / 100.0


def parse_point(point):
    """Parse a point string like 'z050_xm050_y100' into a coordinate dict."""
    parts = point.split("_")
    return {
        "z_m": parse_coord_token(next(p for p in parts if p.startswith("z")), "z"),
        "x_m": parse_coord_token(next(p for p in parts if p.startswith("x")), "x"),
        "y_m": parse_coord_token(next(p for p in parts if p.startswith("y")), "y"),
    }


def parse_stdout(path, trial, point, condition, anchor):
    """
    Parse a ranging stdout file into a DataFrame.
    The 'trial' column distinguishes r1 vs r2 data in the combined dataset.
    """
    rows = []
    current = None

    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()

        if s.startswith("sequence n:"):
            if current and "sequence_n" in current:
                rows.append(current)
            current = {
                "trial": trial,
                "point": point,
                "condition": condition,
                "anchor": anchor,
                **parse_point(point),
            }
            try:
                current["sequence_n"] = int(s.split(":")[-1].strip())
            except Exception:
                current["sequence_n"] = np.nan

        if current is None:
            continue

        def first_float():
            """Extract the first number from the current line."""
            m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
            return float(m.group(0)) if m else np.nan

        if s.startswith("status:"):
            current["ok"] = "Ok (0x0)" in s
        elif s.startswith("distance:"):
            current["distance_cm"] = first_float()
        elif s.startswith("AoA azimuth:"):
            current["aoa_azimuth_deg"] = first_float()
        elif s.startswith("AoA elevation:"):
            current["aoa_elevation_deg"] = first_float()
        elif s.startswith("rssi:"):
            current["rssi_dbm"] = first_float()

    if current and "sequence_n" in current:
        rows.append(current)

    return pd.DataFrame(rows)


def mean_or_nan(vals):
    return float(np.mean(vals)) if vals else np.nan


def parse_diag(path, trial, point, condition, anchor):
    """
    Parse a diagnostic JSON file and compute per-sequence averages of signal metrics.
    Returns one row per ranging sequence.
    """
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception as e:
        print(f"WARNING: failed to read {path}: {e}")
        return pd.DataFrame()

    rows = []
    for obj in data:
        row = {
            "trial": trial,
            "point": point,
            "condition": condition,
            "anchor": anchor,
            **parse_point(point),
            "sequence_n": obj.get("sequence_n"),
        }

        rsl = []
        path1_rsl = []
        path1_snr = []
        peak_snr = []
        aoa0 = []  # AoA type 0 = x-axis
        aoa1 = []  # AoA type 1 = y-axis

        for report in obj.get("reports", []):
            for field in report.get("fields", []):
                for sm in field.get("segment_metrics", []):
                    if "rsl_dbm" in sm:
                        rsl.append(sm["rsl_dbm"])
                    if "path1_rsl_dbm" in sm:
                        path1_rsl.append(sm["path1_rsl_dbm"])
                    if "path1_snr" in sm:
                        path1_snr.append(sm["path1_snr"])
                    if "peak_snr" in sm:
                        peak_snr.append(sm["peak_snr"])

                for a in field.get("aoa", []):
                    if a.get("aoa_type") == 0 and "aoa" in a:
                        aoa0.append(a["aoa"])
                    if a.get("aoa_type") == 1 and "aoa" in a:
                        aoa1.append(a["aoa"])

        row["diag_rsl_dbm_mean"] = mean_or_nan(rsl)
        row["diag_path1_rsl_dbm_mean"] = mean_or_nan(path1_rsl)
        row["diag_path1_snr_mean"] = mean_or_nan(path1_snr)
        row["diag_peak_snr_mean"] = mean_or_nan(peak_snr)
        row["diag_aoa_type0_deg_mean"] = mean_or_nan(aoa0)
        row["diag_aoa_type1_deg_mean"] = mean_or_nan(aoa1)
        rows.append(row)

    return pd.DataFrame(rows)


def load_trial(trial):
    """
    Load and clean all measurement data for one trial (r1 or r2).
    Merges stdout + diagnostic data and filters to valid measurements.
    """
    raw = Path(f"trials/{trial}/raw")
    if not raw.exists():
        raise SystemExit(f"Missing folder: {raw}")

    stdout_frames = []
    diag_frames = []

    for p in raw.glob("*"):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        d = m.groupdict()
        if d["kind"] == "stdout.txt":
            stdout_frames.append(parse_stdout(p, trial, d["point"], d["condition"], d["anchor"]))
        elif d["kind"] == "diag.json":
            diag_frames.append(parse_diag(p, trial, d["point"], d["condition"], d["anchor"]))

    stdout = pd.concat(stdout_frames, ignore_index=True) if stdout_frames else pd.DataFrame()
    diag = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()

    merged = stdout.merge(
        diag,
        on=["trial", "point", "condition", "anchor", "x_m", "y_m", "z_m", "sequence_n"],
        how="left",
    )

    # Filter to valid measurements only (status OK and positive distance)
    merged = merged[(merged["ok"] == True) & (merged["distance_cm"].fillna(0) > 0)].copy()
    return merged.reset_index(drop=True)


print("Loading r1...")
r1 = load_trial("r1")
print("Loading r2...")
r2 = load_trial("r2")

# r2 only covers 4 of the 24 r1 points — filter r1 to those same 4 points for fair comparison
r2_points = sorted(r2["point"].unique())
r1_sub = r1[r1["point"].isin(r2_points)].copy()

# Save the combined dataset (both trials, same 4 points) for use by demo_bayes_r2_table.py
combined = pd.concat([r1_sub, r2], ignore_index=True)
combined.to_csv(OUT / "trials_r1_r2_measurements_clean.csv", index=False)

features = [f for f in FEATURES if f in combined.columns]
train = r1_sub.dropna(subset=features).copy()
test = r2.dropna(subset=features).copy()

X_train = train[features].replace([np.inf, -np.inf], np.nan)
X_test = test[features].replace([np.inf, -np.inf], np.nan)

# Fill any remaining NaNs with the training set medians (not test set — avoids data leakage)
med = X_train.median(numeric_only=True)
X_train = X_train.fillna(med)
X_test = X_test.fillna(med)

# Obstruction classifier: trained on r1, evaluated on r2
clf = RandomForestClassifier(
    n_estimators=500,
    random_state=10,
    min_samples_leaf=5,
    class_weight="balanced",  # Handle any class imbalance
    n_jobs=-1,
)
clf.fit(X_train, train["condition"])
cond_pred = clf.predict(X_test)
cond_acc = accuracy_score(test["condition"], cond_pred)

# 3D localization model: trained on r1, evaluated on r2
reg = ExtraTreesRegressor(
    n_estimators=500,
    random_state=11,
    min_samples_leaf=5,
    n_jobs=-1,
)
reg.fit(X_train, train[["x_m", "y_m", "z_m"]])
pred_xyz = reg.predict(X_test)

true_xyz = test[["x_m", "y_m", "z_m"]].to_numpy(float)
err = np.linalg.norm(pred_xyz - true_xyz, axis=1)  # 3D Euclidean error per measurement

summary = pd.DataFrame([{
    "train_trial": "r1",
    "test_trial": "r2",
    "n_train_rows": len(train),
    "n_test_rows": len(test),
    "n_test_points": len(r2_points),
    "n_features": len(features),
    "condition_accuracy": float(cond_acc),
    "mean_3d_error_m": float(np.mean(err)),
    "median_3d_error_m": float(np.median(err)),
    "p75_3d_error_m": float(np.quantile(err, 0.75)),
    "p90_3d_error_m": float(np.quantile(err, 0.90)),
}])

summary.to_csv(OUT / "trial_generalization_r1_to_r2_results.csv", index=False)

pred_out = test[["trial", "point", "condition", "anchor", "sequence_n", "x_m", "y_m", "z_m"]].copy()
pred_out[["pred_x_m", "pred_y_m", "pred_z_m"]] = pred_xyz
pred_out["error_3d_m"] = err
pred_out["pred_condition"] = cond_pred
pred_out["condition_correct"] = pred_out["pred_condition"] == pred_out["condition"]
pred_out.to_csv(OUT / "trial_generalization_r1_to_r2_predictions.csv", index=False)

# Aggregate per-point summary: how well did the model do at each test location?
per_point = (
    pred_out.groupby(["point", "condition"])
    .agg(
        n=("error_3d_m", "size"),
        mean_error_m=("error_3d_m", "mean"),
        median_error_m=("error_3d_m", "median"),
        condition_acc=("condition_correct", "mean"),
    )
    .reset_index()
    .sort_values(["point", "condition"])
)
per_point.to_csv(OUT / "trial_generalization_r1_to_r2_per_point.csv", index=False)

print()
print("=== Trial generalization r1 -> r2 ===")
print(summary.to_string(index=False))

print()
print("=== R2 points used ===")
for p in r2_points:
    print(" ", p)

print()
print("=== Per-point r2 errors ===")
print(per_point.to_string(index=False))
