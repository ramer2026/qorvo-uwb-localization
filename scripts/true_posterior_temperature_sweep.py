# true_posterior_temperature_sweep.py
#
# Step 5 of the pipeline: find the optimal "temperature" for Bayesian posterior calibration.
#
# Inputs:  processed/measurements_clean.csv  (from analyze_qorvo_complete.py)
# Outputs: processed/true_posterior_temperature_sweep.csv
#
# What is temperature scaling?
#   After computing the raw Bayesian posterior probabilities, we can raise each probability
#   to the power (1/T) and renormalize. This is called temperature scaling:
#     - T = 1.0: original posterior (no change)
#     - T > 1.0: "softer" / flatter distribution — the model becomes less confident
#     - T < 1.0: "sharper" / more peaked distribution — the model becomes more confident
#
#   The Bayesian model is typically overconfident (ECE >> 0), because it assumes features
#   are independent and perfectly Gaussian. Temperature scaling is a simple post-hoc fix
#   that can improve calibration without changing the model architecture.
#
#   We sweep T from 1.0 to 50.0 and record accuracy/calibration at each value.
#   The output CSV lets you choose the temperature that best balances calibration vs accuracy.
#
# Run with: python scripts/true_posterior_temperature_sweep.py

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

PROCESSED = Path("processed")
df = pd.read_csv(PROCESSED / "measurements_clean.csv")

# Full diagnostic feature set — same as latent_bayes_uncertainty.py 'full_diag'
FEATURES = [
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
    "diag_aoa_type0_fom_mean",
    "diag_aoa_type1_fom_mean",
]
FEATURES = [f for f in FEATURES if f in df.columns]

use = df[
    ["point", "condition", "anchor", "sequence_n", "x_m", "y_m", "z_m"] + FEATURES
].copy()

use = use.replace([np.inf, -np.inf], np.nan)
use = use.dropna(subset=FEATURES).reset_index(drop=True)
use["point_condition"] = use["point"] + "__" + use["condition"]

# 70/30 stratified split so each point_condition is represented in train and test
train, test = train_test_split(
    use,
    test_size=0.30,
    random_state=7,
    stratify=use["point_condition"],
)

# Coordinate lookup for error calculation
coord = use.groupby("point")[["x_m", "y_m", "z_m"]].first().to_dict("index")


def fit_models(train):
    """
    Fit one diagonal Gaussian per (point, condition) label.
    Sigma is regularized to avoid degenerate distributions.
    """
    models = {}
    global_sigma = train[FEATURES].std(ddof=1).replace(0, np.nan).fillna(1.0).to_numpy(float)

    for label, g in train.groupby("point_condition"):
        X = g[FEATURES].to_numpy(float)
        mu = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0, ddof=1)

        sigma = np.where(np.isfinite(sigma), sigma, global_sigma)
        sigma = np.maximum(sigma, 0.15 * global_sigma)
        sigma = np.maximum(sigma, 1e-3)

        models[label] = {
            "mu": mu,
            "sigma": sigma,
            "point": label.split("__")[0],
            "condition": label.split("__")[1],
            "n": len(g),
        }
    return models


def logpdf(x, mu, sigma):
    """Log probability of x under a diagonal Gaussian N(mu, sigma)."""
    sigma = np.maximum(sigma, 1e-6)
    z = (x - mu) / sigma
    return -0.5 * np.sum(z*z + np.log(2*np.pi*sigma*sigma))


def posterior_pc(x, models):
    """
    Compute the raw Bayesian posterior (T=1) over all (point, condition) labels for x.
    Uses log-sum-exp trick for numerical stability.
    """
    labels = list(models.keys())
    logps = np.array([logpdf(x, models[l]["mu"], models[l]["sigma"]) for l in labels])
    logps -= np.max(logps)
    probs = np.exp(logps)
    probs /= probs.sum()
    return labels, probs


def apply_temperature(probs, T):
    """
    Apply temperature scaling: raise each probability to power (1/T) and renormalize.
    T > 1 flattens the distribution (less confident), T < 1 sharpens it (more confident).
    """
    p = np.asarray(probs, dtype=float)
    p = np.maximum(p, 1e-300)  # Prevent log(0) issues
    p = p ** (1.0 / T)
    p /= p.sum()
    return p


def expected_calibration_error(conf, correct, n_bins=10):
    """
    ECE: average gap between confidence and accuracy across probability bins.
    Lower ECE = better calibrated.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)

    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        if not mask.any():
            continue

        ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())

    return float(ece)


def credible_metrics(point_probs, true_point, level=0.95):
    """
    Compute 95% credible set coverage and size.
    Returns: (covered: bool, set_size: int, max_radius_m: float)
    """
    items = sorted(point_probs.items(), key=lambda kv: kv[1], reverse=True)

    cum = 0.0
    included = []
    for point, prob in items:
        included.append(point)
        cum += prob
        if cum >= level:
            break

    best = items[0][0]
    best_xyz = np.array([coord[best]["x_m"], coord[best]["y_m"], coord[best]["z_m"]])

    radii = []
    for p in included:
        xyz = np.array([coord[p]["x_m"], coord[p]["y_m"], coord[p]["z_m"]])
        radii.append(float(np.linalg.norm(xyz - best_xyz)))

    return true_point in included, len(included), max(radii) if radii else 0.0


# Train the Gaussian models once on the training set
models = fit_models(train)

# Pre-compute raw posteriors (T=1) for every test row — we'll reuse these for all T values
# This avoids re-running the Gaussian inference 16 times
base_records = []

for _, r in test.iterrows():
    x = r[FEATURES].to_numpy(float)
    labels, pc_probs = posterior_pc(x, models)

    base_records.append({
        "true_point": r["point"],
        "true_condition": r["condition"],
        "x_m": r["x_m"],
        "y_m": r["y_m"],
        "z_m": r["z_m"],
        "labels": labels,
        "pc_probs": pc_probs,  # Raw T=1 probabilities, reused for all temperatures
    })

rows = []
prediction_rows = []

# Sweep temperatures from near-original (1.0) to very flat (50.0)
for T in [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0]:
    point_correct = []
    top3_correct = []
    top5_correct = []
    condition_correct = []
    errors = []
    confs = []
    cond_confs = []
    prob_obsts = []
    true_obsts = []
    cover95 = []
    size95 = []
    radius95 = []

    for rec in base_records:
        labels = rec["labels"]
        # Apply temperature scaling to the stored raw posterior
        pc_probs = apply_temperature(rec["pc_probs"], T)

        # Marginalize: sum over conditions to get per-point probabilities
        point_probs = {}
        cond_probs = {"los": 0.0, "obst": 0.0}

        for lab, prob in zip(labels, pc_probs):
            point = models[lab]["point"]
            cond = models[lab]["condition"]
            point_probs[point] = point_probs.get(point, 0.0) + float(prob)
            cond_probs[cond] = cond_probs.get(cond, 0.0) + float(prob)

        sorted_points = sorted(point_probs.items(), key=lambda kv: kv[1], reverse=True)
        pred_point = sorted_points[0][0]
        pred_point_prob = sorted_points[0][1]

        pred_condition = max(cond_probs.items(), key=lambda kv: kv[1])[0]
        pred_condition_prob = cond_probs[pred_condition]

        true_xyz = np.array([rec["x_m"], rec["y_m"], rec["z_m"]], dtype=float)
        pred_xyz = np.array([
            coord[pred_point]["x_m"],
            coord[pred_point]["y_m"],
            coord[pred_point]["z_m"],
        ], dtype=float)
        err = float(np.linalg.norm(pred_xyz - true_xyz))

        covered, size, radius = credible_metrics(point_probs, rec["true_point"], level=0.95)

        point_correct.append(pred_point == rec["true_point"])
        top3_correct.append(rec["true_point"] in [p for p, _ in sorted_points[:3]])
        top5_correct.append(rec["true_point"] in [p for p, _ in sorted_points[:5]])
        condition_correct.append(pred_condition == rec["true_condition"])
        errors.append(err)
        confs.append(pred_point_prob)
        cond_confs.append(pred_condition_prob)
        prob_obsts.append(cond_probs["obst"])
        true_obsts.append(1 if rec["true_condition"] == "obst" else 0)
        cover95.append(covered)
        size95.append(size)
        radius95.append(radius)

    rows.append({
        "temperature": T,
        "point_top1_accuracy": float(np.mean(point_correct)),
        "point_top3_accuracy": float(np.mean(top3_correct)),
        "point_top5_accuracy": float(np.mean(top5_correct)),
        "condition_accuracy": float(np.mean(condition_correct)),
        "mean_3d_error_m": float(np.mean(errors)),
        "median_3d_error_m": float(np.median(errors)),
        "p75_3d_error_m": float(np.quantile(errors, 0.75)),
        "p90_3d_error_m": float(np.quantile(errors, 0.90)),
        "credible95_coverage": float(np.mean(cover95)),
        "mean_credible95_size": float(np.mean(size95)),
        "mean_credible95_radius_m": float(np.mean(radius95)),
        "mean_best_point_confidence": float(np.mean(confs)),
        "point_ece": expected_calibration_error(confs, point_correct),
        "condition_ece": expected_calibration_error(cond_confs, condition_correct),
        "condition_brier_obst": float(brier_score_loss(true_obsts, prob_obsts)),
    })

out = pd.DataFrame(rows)
out.to_csv(PROCESSED / "true_posterior_temperature_sweep.csv", index=False)

print(out.to_string(index=False))
print("\nWrote processed/true_posterior_temperature_sweep.csv")
