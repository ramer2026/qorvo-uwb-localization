# aggregate_fingerprint_model.py
#
# Step 3 of the pipeline: fingerprint-based localization using aggregated statistics.
#
# Inputs:  processed/measurements_clean.csv  (from analyze_qorvo_complete.py)
# Outputs: processed/aggregate_fingerprint_table.csv   — one row per (point, condition)
#          processed/aggregate_localization_results.csv — 3D error for each feature set
#          processed/aggregate_predictions_*.csv        — per-row predictions
#          processed/aggregate_obstruction_classifier_results.csv
#
# "Fingerprinting" means instead of using raw individual measurements, we summarize
# each location's signal behavior as a set of statistics (mean, std, IQR, etc.).
# The idea is: every room location has a unique statistical "fingerprint" in signal space.
# The model learns to map these fingerprints back to 3D coordinates.
#
# Run with: python scripts/aggregate_fingerprint_model.py

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix


PROCESSED = Path("processed")
df = pd.read_csv(PROCESSED / "measurements_clean.csv")

# These are the raw signal features we compute statistics over
features = [
    "distance_cm",          # UWB-measured range to anchor (cm)
    "rssi_dbm",             # Received Signal Strength Indicator (dBm; more negative = weaker)
    "aoa_azimuth_deg",      # Horizontal angle of signal arrival (degrees)
    "aoa_elevation_deg",    # Vertical angle of signal arrival (degrees)
    "diag_rsl_dbm_mean",    # Average Received Signal Level from diagnostic segments
    "diag_path1_rsl_dbm_mean",  # First-path RSL (direct line-of-sight component)
    "diag_path1_snr_mean",      # First-path Signal-to-Noise Ratio
    "diag_peak_snr_mean",       # Peak SNR across all paths
    "diag_aoa_type0_deg_mean",  # Raw AoA on x-axis from diagnostics
    "diag_aoa_type1_deg_mean",  # Raw AoA on y-axis from diagnostics
]

features = [f for f in features if f in df.columns]


# Distribution summary functions — these create the "fingerprint" for each location
def iqr(x):
    """Interquartile range: Q75 - Q25. Robust measure of spread, less sensitive to outliers."""
    x = np.asarray(x.dropna(), dtype=float)
    if len(x) == 0:
        return np.nan
    return np.quantile(x, 0.75) - np.quantile(x, 0.25)

def q10(x):
    """10th percentile of the distribution."""
    x = np.asarray(x.dropna(), dtype=float)
    return np.quantile(x, 0.10) if len(x) else np.nan

def q90(x):
    """90th percentile of the distribution."""
    x = np.asarray(x.dropna(), dtype=float)
    return np.quantile(x, 0.90) if len(x) else np.nan


# Build the fingerprint table: one row per (point, condition) with 6 statistics per feature
rows = []

for (point, condition), gpc in df.groupby(["point", "condition"]):
    row = {
        "point": point,
        "condition": condition,
        "is_obst": 1 if condition == "obst" else 0,  # Binary label for obstruction classifier
        "x_m": gpc["x_m"].iloc[0],
        "y_m": gpc["y_m"].iloc[0],
        "z_m": gpc["z_m"].iloc[0],
        "n_total": len(gpc),
    }

    # Compute statistics separately per anchor (A and B) to preserve spatial information
    for anchor, ga in gpc.groupby("anchor"):
        row[f"{anchor}_n"] = len(ga)

        for feat in features:
            vals = ga[feat].dropna().astype(float)
            # Each feature becomes 6 columns: mean, std, median, IQR, Q10, Q90
            row[f"{anchor}_{feat}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{anchor}_{feat}_std"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
            row[f"{anchor}_{feat}_median"] = vals.median() if len(vals) else np.nan
            row[f"{anchor}_{feat}_iqr"] = iqr(vals)
            row[f"{anchor}_{feat}_q10"] = q10(vals)
            row[f"{anchor}_{feat}_q90"] = q90(vals)

    rows.append(row)

agg = pd.DataFrame(rows)
agg.to_csv(PROCESSED / "aggregate_fingerprint_table.csv", index=False)

# All numeric feature columns (excludes coordinates, labels, and counts)
numeric_features = [
    c for c in agg.columns
    if c not in ["point", "condition", "x_m", "y_m", "z_m", "is_obst"]
    and pd.api.types.is_numeric_dtype(agg[c])
]


def eval_regression(name, feat_filter):
    """
    Train and evaluate an ExtraTrees regression model for 3D localization.

    feat_filter: a lambda that returns True for column names to include as features.
    Uses GroupKFold cross-validation grouped by point so the model is tested on
    locations it has never seen during training (realistic deployment simulation).

    ExtraTrees (Extremely Randomized Trees) is used here instead of Random Forest
    because it's faster and performs similarly on tabular data at this scale.
    """
    feats = [c for c in numeric_features if feat_filter(c)]
    X = agg[feats].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = agg[["x_m", "y_m", "z_m"]].to_numpy(float)
    groups = agg["point"].astype(str).to_numpy()

    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    preds = np.zeros_like(y)

    for train, test in cv.split(X, y, groups):
        model = ExtraTreesRegressor(
            n_estimators=800,
            random_state=11,
            min_samples_leaf=1,
            n_jobs=-1,
        )
        model.fit(X.iloc[train], y[train])
        preds[test] = model.predict(X.iloc[test])

    # 3D Euclidean error: distance between predicted and true location
    err = np.linalg.norm(preds - y, axis=1)

    pred = agg[["point", "condition", "x_m", "y_m", "z_m"]].copy()
    pred[["pred_x_m", "pred_y_m", "pred_z_m"]] = preds
    pred["error_3d_m"] = err
    pred.to_csv(PROCESSED / f"aggregate_predictions_{name}.csv", index=False)

    return {
        "model": name,
        "n_samples": len(agg),
        "n_features": len(feats),
        "mean_3d_error_m": float(np.mean(err)),
        "median_3d_error_m": float(np.median(err)),
        "p75_3d_error_m": float(np.quantile(err, 0.75)),
        "p90_3d_error_m": float(np.quantile(err, 0.90)),
        "rmse_x_m": math.sqrt(mean_squared_error(y[:,0], preds[:,0])),
        "rmse_y_m": math.sqrt(mean_squared_error(y[:,1], preds[:,1])),
        "rmse_z_m": math.sqrt(mean_squared_error(y[:,2], preds[:,2])),
    }


reg_rows = []

# Model 1: Only use range and AoA statistics (minimal feature set)
reg_rows.append(eval_regression(
    "agg_range_aoa_only",
    lambda c: any(k in c for k in ["distance_cm", "aoa_azimuth", "aoa_elevation"])
))

# Model 2: Add RSSI and signal quality statistics
reg_rows.append(eval_regression(
    "agg_range_aoa_signal_snr",
    lambda c: any(k in c for k in ["distance_cm", "aoa_azimuth", "aoa_elevation", "rssi", "rsl", "snr"])
))

# Model 3: All distribution statistics for all features (full fingerprint)
reg_rows.append(eval_regression(
    "agg_all_distribution_features",
    lambda c: True
))

reg_results = pd.DataFrame(reg_rows).sort_values("mean_3d_error_m")
reg_results.to_csv(PROCESSED / "aggregate_localization_results.csv", index=False)


# Obstruction classifier using aggregate/distribution features
# Predict LOS vs obstructed from the statistical fingerprint alone
X = agg[numeric_features].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
y = agg["is_obst"].to_numpy(int)
groups = agg["point"].astype(str).to_numpy()

cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
preds = np.zeros_like(y)

for train, test in cv.split(X, y, groups):
    clf = RandomForestClassifier(
        n_estimators=800,
        random_state=12,
        min_samples_leaf=1,
        class_weight="balanced",  # Balance LOS/obstructed class weights since they may not be equal
        n_jobs=-1,
    )
    clf.fit(X.iloc[train], y[train])
    preds[test] = clf.predict(X.iloc[test])

acc = accuracy_score(y, preds)
report = classification_report(y, preds, target_names=["los", "obst"], output_dict=True)
cm = confusion_matrix(y, preds)

clf_results = pd.DataFrame([{
    "classifier": "aggregate_distribution_random_forest",
    "n_samples": len(agg),
    "n_features": len(numeric_features),
    "accuracy": acc,
    "los_precision": report["los"]["precision"],
    "los_recall": report["los"]["recall"],
    "obst_precision": report["obst"]["precision"],
    "obst_recall": report["obst"]["recall"],
    # Confusion matrix breakdown
    "tn_los_correct": cm[0,0],
    "fp_los_as_obst": cm[0,1],
    "fn_obst_as_los": cm[1,0],
    "tp_obst_correct": cm[1,1],
}])

clf_results.to_csv(PROCESSED / "aggregate_obstruction_classifier_results.csv", index=False)

print("\n=== Aggregate localization results ===")
print(reg_results.to_string(index=False))

print("\n=== Aggregate obstruction classifier ===")
print(clf_results.to_string(index=False))

print("\nWrote:")
print("  processed/aggregate_fingerprint_table.csv")
print("  processed/aggregate_localization_results.csv")
print("  processed/aggregate_obstruction_classifier_results.csv")
