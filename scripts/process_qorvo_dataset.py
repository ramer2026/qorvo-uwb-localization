# process_qorvo_dataset.py
#
# Step 1 of the pipeline: parse raw Qorvo UWB measurement files into clean CSVs.
#
# Inputs:  raw/  folder containing .txt (ranging output) and .json (diagnostics) files
# Outputs: processed/  folder with several CSV files, most importantly:
#            - measurements_clean.csv  (the main dataset used by all other scripts)
#            - los_vs_obst_stats.csv   (statistical comparison of LOS vs obstructed signal)
#
# Run with: python scripts/process_qorvo_dataset.py
#           or: python scripts/process_qorvo_dataset.py --raw-dir raw --out-dir processed

import argparse
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# Regex that matches the naming convention of every raw file.
# Example filename: z050_xm050_y100_los_anchorA_stdout.txt
#   z050  = height 0.50m, xm050 = x=-0.50m, y100 = y=1.00m
#   los   = line-of-sight (clear path), obst = obstructed
#   anchorA/anchorB = which fixed radio beacon picked up the signal
#   stdout/diag = ranging text log vs JSON diagnostic data
FILENAME_RE = re.compile(
    r"^(?P<point>(?P<z>z(?:m)?\d{3})_(?P<x>x(?:m)?\d{3})_(?P<y>y(?:m)?\d{3}))_"
    r"(?P<condition>los|obst)_(?P<anchor>anchorA|anchorB)_(?P<kind>stdout|diag)\.(?P<ext>txt|json)$"
)


def coord_token_to_m(token: str) -> float:
    """
    Convert a coordinate token from the filename into meters.
    The 'm' prefix means negative; the 3-digit number is centimeters * 10.
    Examples:
      z050  -> +0.50
      z000  ->  0.00
      zm050 -> -0.50
      xm050 -> -0.50
      y100  -> +1.00
    """
    axis = token[0]
    rest = token[1:]
    sign = 1.0

    if rest.startswith("m"):
        sign = -1.0
        rest = rest[1:]

    return sign * (int(rest) / 100.0)


def parse_filename(path: Path):
    """Extract metadata (coordinates, condition, anchor, kind) from a raw filename."""
    m = FILENAME_RE.match(path.name)
    if not m:
        return None

    d = m.groupdict()
    z_m = coord_token_to_m(d["z"])
    x_m = coord_token_to_m(d["x"])
    y_m = coord_token_to_m(d["y"])

    return {
        "point": d["point"],
        "condition": d["condition"],
        "anchor": d["anchor"],
        "kind": d["kind"],
        "x_m": x_m,
        "y_m": y_m,
        "z_m": z_m,
    }


def true_range_cm(x_m, y_m, z_m, fixed_x=0.0, fixed_y=0.5, fixed_z=0.0):
    """
    Compute the true Euclidean distance (in cm) from a measurement point to the anchor.
    The anchor is fixed at (0.0, 0.5, 0.0) meters in the room coordinate system.
    This is the ground-truth distance we compare against the UWB-measured distance.
    """
    return 100.0 * math.sqrt(
        (x_m - fixed_x) ** 2
        + (y_m - fixed_y) ** 2
        + (z_m - fixed_z) ** 2
    )


def maybe_float(s):
    """Parse a string to float; return NaN if it fails."""
    try:
        return float(s)
    except Exception:
        return np.nan


def maybe_int(s):
    """Parse a string to int; return NaN if it fails."""
    try:
        return int(s)
    except Exception:
        return np.nan


def parse_stdout_file(path: Path):
    """
    Parse a Qorvo ranging stdout .txt file into a list of measurement dicts.

    Each measurement block in the file starts with "# Ranging Data:" and contains
    fields like distance, AoA (angle of arrival), RSSI (signal strength), etc.
    We use regex to extract each field line-by-line.

    Returns a list of dicts, one per ranging measurement in the file.
    """
    meta = parse_filename(path)
    if not meta:
        return []

    rows = []
    cur = None  # Accumulates fields for the current ranging block

    def finish_cur():
        """Finalize the current measurement block and append it to rows."""
        nonlocal cur
        if cur is not None and "sequence_n" in cur:
            # A measurement is 'valid' if status is OK and distance is positive
            cur["valid"] = (
                cur.get("status", "") == "Ok (0x0)"
                and pd.notna(cur.get("distance_cm", np.nan))
                and cur.get("distance_cm", 0) > 0
            )
            cur["true_range_cm"] = true_range_cm(cur["x_m"], cur["y_m"], cur["z_m"])
            # Range error = how far off the UWB measurement was from ground truth
            cur["range_error_cm"] = (
                cur["distance_cm"] - cur["true_range_cm"]
                if pd.notna(cur.get("distance_cm", np.nan))
                else np.nan
            )
            rows.append(cur)
        cur = None

    with path.open("r", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # Each ranging block starts with this header line
            if line.startswith("# Ranging Data:"):
                finish_cur()
                cur = {
                    **meta,
                    "source_file": path.name,
                }
                continue

            if cur is None:
                continue

            # Extract each field using regex patterns matching the Qorvo output format
            m = re.search(r"sequence n:\s+(\d+)", line)
            if m:
                cur["sequence_n"] = maybe_int(m.group(1))
                continue

            m = re.search(r"ranging interval:\s+([-\d.]+)\s*ms", line)
            if m:
                cur["ranging_interval_ms"] = maybe_float(m.group(1))
                continue

            m = re.search(r"measurement type:\s+(.+)$", line)
            if m:
                cur["measurement_type"] = m.group(1).strip()
                continue

            m = re.search(r"status:\s+(.+)$", line)
            if m:
                cur["status"] = m.group(1).strip()
                continue

            m = re.search(r"mac address:\s+(.+)$", line)
            if m:
                cur["mac_address"] = m.group(1).strip()
                continue

            m = re.search(r"distance:\s+([-\d.]+)\s*cm", line)
            if m:
                cur["distance_cm"] = maybe_float(m.group(1))
                continue

            # AoA (Angle of Arrival): the angle the signal came from, in degrees
            m = re.search(r"AoA azimuth:\s+([-\d.]+)\s*deg", line)
            if m:
                cur["aoa_azimuth_deg"] = maybe_float(m.group(1))
                continue

            # FOM = Figure of Merit: confidence score for the AoA measurement (0-100%)
            m = re.search(r"AoA az\. FOM:\s+([-\d.]+)\s*%", line)
            if m:
                cur["aoa_az_fom_pct"] = maybe_float(m.group(1))
                continue

            m = re.search(r"AoA elevation:\s+([-\d.]+)\s*deg", line)
            if m:
                cur["aoa_elevation_deg"] = maybe_float(m.group(1))
                continue

            m = re.search(r"AoA elev\. FOM:\s+([-\d.]+)\s*%", line)
            if m:
                cur["aoa_elev_fom_pct"] = maybe_float(m.group(1))
                continue

            # "Dest" AoA = angle from the perspective of the destination device
            m = re.search(r"AoA dest azimuth:\s+([-\d.]+)\s*deg", line)
            if m:
                cur["aoa_dest_azimuth_deg"] = maybe_float(m.group(1))
                continue

            m = re.search(r"AoA dest elevation:\s+([-\d.]+)\s*deg", line)
            if m:
                cur["aoa_dest_elevation_deg"] = maybe_float(m.group(1))
                continue

            # RSSI = Received Signal Strength Indicator: how strong the signal was (dBm)
            m = re.search(r"rssi:\s+([-\d.]+)\s*dBm", line)
            if m:
                cur["rssi_dbm"] = maybe_float(m.group(1))
                continue

    finish_cur()
    return rows


def iter_dict_fields(fields):
    """Yield only dict items from a list (diagnostic JSON fields can be mixed types)."""
    for field in fields:
        if isinstance(field, dict):
            yield field


def parse_diag_file(path: Path):
    """
    Parse a Qorvo diagnostic .json file into two lists of dicts:
      - segment_rows: low-level signal metrics per antenna segment (RSL, SNR, CFO)
      - aoa_rows: raw angle-of-arrival measurements per axis (x/y/z)

    These provide richer signal-quality information than the stdout file alone.
    RSL = Received Signal Level (dBm), SNR = Signal-to-Noise Ratio, CFO = Carrier Frequency Offset
    """
    meta = parse_filename(path)
    if not meta:
        return [], []

    with path.open("r", errors="ignore") as f:
        data = json.load(f)

    segment_rows = []
    aoa_rows = []

    if not isinstance(data, list):
        return segment_rows, aoa_rows

    # Each entry in the JSON list corresponds to one ranging sequence
    for seq_entry in data:
        sequence_n = seq_entry.get("sequence_n")
        session_handle = seq_entry.get("session_handle")

        for report in seq_entry.get("reports", []):
            report_n = report.get("report_n")
            msg_id = report.get("msg_id")
            action = report.get("action")
            antenna_set = report.get("antenna_set")

            cfo_val = np.nan

            for field in iter_dict_fields(report.get("fields", [])):
                # CFO indicates Doppler/frequency shift — proxy for multipath or motion
                if "cfo" in field:
                    cfo_val = field.get("cfo", np.nan)

                if "segment_metrics" in field:
                    for sm in field.get("segment_metrics", []):
                        row = {
                            **meta,
                            "source_file": path.name,
                            "session_handle": session_handle,
                            "sequence_n": sequence_n,
                            "report_n": report_n,
                            "msg_id": msg_id,
                            "action": action,
                            "antenna_set": antenna_set,
                            "cfo": cfo_val,
                        }
                        row.update(sm)
                        segment_rows.append(row)

                if "aoa" in field:
                    for aoa in field.get("aoa", []):
                        row = {
                            **meta,
                            "source_file": path.name,
                            "session_handle": session_handle,
                            "sequence_n": sequence_n,
                            "report_n": report_n,
                            "msg_id": msg_id,
                            "action": action,
                            "antenna_set": antenna_set,
                        }
                        row.update(aoa)
                        aoa_rows.append(row)

    return segment_rows, aoa_rows


def aggregate_diag(segment_df, aoa_df):
    """
    Collapse diagnostic segment and AoA data from multiple reports per sequence
    into one summary row per (point, condition, anchor, sequence_n).

    This aligns diagnostic features with the stdout ranging measurements so they
    can be merged together into one flat row per measurement.
    """
    parts = []

    keys = ["point", "condition", "anchor", "sequence_n"]

    if not segment_df.empty:
        # Average segment metrics across all antenna segments in one ranging exchange
        agg_seg = (
            segment_df
            .groupby(keys, dropna=False)
            .agg(
                rsl_dbm_mean=("rsl_dbm", "mean"),
                rsl_dbm_std=("rsl_dbm", "std"),
                path1_rsl_dbm_mean=("path1_rsl_dbm", "mean"),
                path1_snr_mean=("path1_snr", "mean"),
                path1_snr_std=("path1_snr", "std"),
                peak_rsl_dbm_mean=("peak_rsl_dbm", "mean"),
                peak_snr_mean=("peak_snr", "mean"),
                peak_snr_std=("peak_snr", "std"),
                cfo_mean=("cfo", "mean"),
                n_segment_metrics=("rsl_dbm", "count"),
            )
            .reset_index()
        )
        parts.append(agg_seg)

    if not aoa_df.empty:
        # AoA type 0/1/2 correspond to x/y/z axes — pivot them into separate columns
        aoa_wide_parts = []

        for aoa_type, label in [(0, "x"), (1, "y"), (2, "z")]:
            tmp = aoa_df[aoa_df["aoa_type"] == aoa_type]
            if tmp.empty:
                continue

            tmp_agg = (
                tmp
                .groupby(keys, dropna=False)
                .agg(
                    **{
                        f"raw_aoa_{label}_deg": ("aoa", "mean"),
                        f"raw_aoa_{label}_std_deg": ("aoa", "std"),
                        f"raw_pdoa_{label}_deg": ("pdoa", "mean"),
                        f"raw_pdoa_{label}_std_deg": ("pdoa", "std"),
                        f"raw_tdoa_{label}_mean": ("tdoa", "mean"),
                        f"raw_aoa_{label}_fom": ("aoa_fom", "mean"),
                        f"n_raw_aoa_{label}": ("aoa", "count"),
                    }
                )
                .reset_index()
            )
            aoa_wide_parts.append(tmp_agg)

        if aoa_wide_parts:
            aoa_features = aoa_wide_parts[0]
            for p in aoa_wide_parts[1:]:
                aoa_features = aoa_features.merge(p, on=keys, how="outer")
            parts.append(aoa_features)

    if not parts:
        return pd.DataFrame(columns=keys)

    # Merge all feature groups together on the shared key columns
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=keys, how="outer")

    return out


def completeness_report(raw_dir: Path):
    """
    Check whether every expected file is present for each point/condition/anchor/kind.
    We expect exactly 4 files per (point, condition): anchorA stdout, anchorA diag,
    anchorB stdout, anchorB diag.
    Returns the full inventory DataFrame and a completeness check DataFrame.
    """
    records = []
    for f in raw_dir.iterdir():
        meta = parse_filename(f)
        if meta:
            records.append({**meta, "filename": f.name})

    df = pd.DataFrame(records)
    if df.empty:
        return df, pd.DataFrame()

    count = (
        df
        .groupby(["point", "condition", "anchor", "kind"])
        .size()
        .reset_index(name="count")
    )

    expected_points = sorted(df["point"].unique())
    expected_conditions = ["los", "obst"]
    expected_anchors = ["anchorA", "anchorB"]
    expected_kinds = ["stdout", "diag"]

    rows = []
    for point in expected_points:
        for cond in expected_conditions:
            for anchor in expected_anchors:
                for kind in expected_kinds:
                    n = count[
                        (count["point"] == point)
                        & (count["condition"] == cond)
                        & (count["anchor"] == anchor)
                        & (count["kind"] == kind)
                    ]["count"]
                    rows.append({
                        "point": point,
                        "condition": cond,
                        "anchor": anchor,
                        "kind": kind,
                        "count": int(n.iloc[0]) if len(n) else 0,
                        "complete": int(n.iloc[0]) == 1 if len(n) else False,
                    })

    complete_df = pd.DataFrame(rows)
    return df, complete_df


def cohen_d(a, b):
    """
    Compute Cohen's d effect size between two samples.
    Measures how many standard deviations apart the two group means are.
    |d| < 0.2 = small, 0.5 = medium, 0.8+ = large effect.
    Used to quantify how much LOS vs obstructed conditions differ for each metric.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan

    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = math.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return (np.mean(b) - np.mean(a)) / pooled


def los_vs_obst_stats(df):
    """
    For each (point, anchor, metric) combination, compute statistics comparing
    LOS measurements against obstructed measurements.

    Uses Welch's t-test (doesn't assume equal variances) to test whether the
    difference between LOS and obstructed distributions is statistically significant.
    Cohen's d tells us how large the practical difference is.

    This is the core analysis that tells us which signal features are most
    affected by obstruction.
    """
    metrics = [
        "distance_cm",
        "range_error_cm",
        "rssi_dbm",
        "aoa_azimuth_deg",
        "aoa_elevation_deg",
        "aoa_dest_azimuth_deg",
        "aoa_dest_elevation_deg",
        "rsl_dbm_mean",
        "path1_snr_mean",
        "peak_snr_mean",
        "raw_aoa_x_deg",
        "raw_aoa_y_deg",
        "raw_aoa_x_fom",
        "raw_aoa_y_fom",
    ]

    rows = []

    valid_df = df.copy()
    if "valid" in valid_df.columns:
        valid_df = valid_df[valid_df["valid"] == True]

    for (point, anchor), g in valid_df.groupby(["point", "anchor"]):
        los = g[g["condition"] == "los"]
        obst = g[g["condition"] == "obst"]

        if los.empty or obst.empty:
            continue

        for metric in metrics:
            if metric not in g.columns:
                continue

            a = pd.to_numeric(los[metric], errors="coerce").dropna()
            b = pd.to_numeric(obst[metric], errors="coerce").dropna()

            if len(a) < 2 or len(b) < 2:
                continue

            if HAVE_SCIPY:
                try:
                    test = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                    p_value = float(test.pvalue)
                    t_stat = float(test.statistic)
                except Exception:
                    p_value = np.nan
                    t_stat = np.nan
            else:
                p_value = np.nan
                t_stat = np.nan

            rows.append({
                "point": point,
                "anchor": anchor,
                "metric": metric,
                "n_los": len(a),
                "n_obst": len(b),
                "mean_los": float(a.mean()),
                "mean_obst": float(b.mean()),
                "median_los": float(a.median()),
                "median_obst": float(b.median()),
                "std_los": float(a.std(ddof=1)),
                "std_obst": float(b.std(ddof=1)),
                "delta_mean_obst_minus_los": float(b.mean() - a.mean()),
                "delta_median_obst_minus_los": float(b.median() - a.median()),
                "cohen_d_obst_minus_los": cohen_d(a, b),
                "welch_t": t_stat,
                "p_value": p_value,
                "significant_0p05": bool(p_value < 0.05) if np.isfinite(p_value) else False,
            })

    return pd.DataFrame(rows)


def point_summary(df):
    """
    Compute summary statistics (mean, std, median) per (point, condition, anchor).
    This creates one row per measurement location/condition/anchor combination,
    which is the format used by the ML models in analyze_qorvo_complete.py.
    """
    valid_df = df.copy()
    if "valid" in valid_df.columns:
        valid_df = valid_df[valid_df["valid"] == True]

    metrics = {
        "distance_cm": ["count", "mean", "std", "median"],
        "range_error_cm": ["mean", "std", "median"],
        "rssi_dbm": ["mean", "std", "median"],
        "aoa_azimuth_deg": ["mean", "std", "median"],
        "aoa_elevation_deg": ["mean", "std", "median"],
        "rsl_dbm_mean": ["mean", "std", "median"],
        "path1_snr_mean": ["mean", "std", "median"],
        "peak_snr_mean": ["mean", "std", "median"],
        "raw_aoa_x_deg": ["mean", "std", "median"],
        "raw_aoa_y_deg": ["mean", "std", "median"],
    }

    available = {k: v for k, v in metrics.items() if k in valid_df.columns}
    if not available:
        return pd.DataFrame()

    out = valid_df.groupby(["point", "condition", "anchor"]).agg(available)
    out.columns = ["_".join(c).strip() for c in out.columns.to_flat_index()]
    out = out.reset_index()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw", help="Folder containing raw files")
    ap.add_argument("--out-dir", default="processed", help="Folder to write CSV outputs")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Inventory all raw files and check for missing files
    files_df, complete_df = completeness_report(raw_dir)
    files_df.to_csv(out_dir / "file_inventory.csv", index=False)
    complete_df.to_csv(out_dir / "file_completeness.csv", index=False)

    stdout_rows = []
    segment_rows = []
    aoa_rows = []

    # Step 2: Parse all recognized files
    for path in sorted(raw_dir.iterdir()):
        meta = parse_filename(path)
        if not meta:
            continue

        if meta["kind"] == "stdout":
            stdout_rows.extend(parse_stdout_file(path))
        elif meta["kind"] == "diag":
            seg, aoa = parse_diag_file(path)
            segment_rows.extend(seg)
            aoa_rows.extend(aoa)

    stdout_df = pd.DataFrame(stdout_rows)
    segment_df = pd.DataFrame(segment_rows)
    aoa_df = pd.DataFrame(aoa_rows)

    # Step 3: Save raw parsed data before any merging
    stdout_df.to_csv(out_dir / "ranging_stdout_rows.csv", index=False)
    segment_df.to_csv(out_dir / "diag_segment_rows.csv", index=False)
    aoa_df.to_csv(out_dir / "diag_aoa_rows.csv", index=False)

    # Step 4: Aggregate diagnostic data and merge with ranging data
    diag_features = aggregate_diag(segment_df, aoa_df)
    diag_features.to_csv(out_dir / "diag_sequence_features.csv", index=False)

    if not stdout_df.empty and not diag_features.empty:
        # Left join: keep all ranging rows, attach diagnostic features where available
        merged = stdout_df.merge(
            diag_features,
            on=["point", "condition", "anchor", "sequence_n"],
            how="left",
        )
    else:
        merged = stdout_df.copy()

    merged.to_csv(out_dir / "merged_measurements.csv", index=False)

    # Step 5: Summarize and compare LOS vs obstructed
    summary = point_summary(merged)
    summary.to_csv(out_dir / "point_condition_summary.csv", index=False)

    stats_df = los_vs_obst_stats(merged)
    stats_df.to_csv(out_dir / "los_vs_obst_stats.csv", index=False)

    print("Done.")
    print(f"Raw dir:       {raw_dir}")
    print(f"Output dir:    {out_dir}")
    print(f"Inventory:     {len(files_df)} recognized files")
    print(f"Stdout rows:   {len(stdout_df)}")
    print(f"Segment rows:  {len(segment_df)}")
    print(f"AoA rows:      {len(aoa_df)}")
    print(f"Merged rows:   {len(merged)}")

    if not complete_df.empty:
        missing = complete_df[complete_df["complete"] == False]
        print(f"Missing expected file slots: {len(missing)}")
        if len(missing):
            print(missing.head(20).to_string(index=False))

    if not stdout_df.empty:
        valid = stdout_df[stdout_df["valid"] == True]
        print(f"Valid stdout measurements: {len(valid)} / {len(stdout_df)}")


if __name__ == "__main__":
    main()
