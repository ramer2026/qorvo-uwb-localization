# latent_bayes_uncertainty.py
#
# Step 4 of the pipeline: Bayesian localization with uncertainty quantification.
#
# Inputs:  processed/measurements_clean.csv  (from analyze_qorvo_complete.py)
# Outputs: processed/latent_bayes_uncertainty_results.csv    — accuracy/error/calibration per feature set
#          processed/latent_bayes_uncertainty_predictions.csv — per-measurement predictions
#          processed/latent_bayes_calibration_bins.csv        — calibration curve data
#
# How it works:
#   For each known location+condition, we fit a multivariate Gaussian (normal distribution)
#   over the signal features in the training data. At test time, we compute how likely a
#   new measurement is under each Gaussian, then use Bayes' rule to get a probability
#   distribution over all possible locations. The highest-probability location is the prediction.
#
#   "Uncertainty" is quantified via:
#     - Posterior entropy: how spread out the probabilities are (high = uncertain)
#     - 95% credible set: the smallest set of candidate locations covering 95% probability mass
#     - ECE (Expected Calibration Error): whether the model's confidence matches its accuracy
#
# Run with: python scripts/latent_bayes_uncertainty.py

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, accuracy_score

PROCESSED = Path("processed")
PLOTS = Path("plots_deeper")
PLOTS.mkdir(exist_ok=True)

df = pd.read_csv(PROCESSED / "measurements_clean.csv")

# Three feature sets to compare — each one includes more signal information
FEATURE_SETS = {
    # Minimal: only the ranging distance and printed angles
    "range_aoa": [
        "distance_cm",
        "aoa_azimuth_deg",
        "aoa_elevation_deg",
    ],
    # Medium: also includes RSSI and signal quality metrics from diagnostics
    "range_aoa_signal_snr": [
        "distance_cm",
        "aoa_azimuth_deg",
        "aoa_elevation_deg",
        "rssi_dbm",
        "diag_rsl_dbm_mean",
        "diag_path1_rsl_dbm_mean",
        "diag_path1_snr_mean",
        "diag_peak_snr_mean",
    ],
    # Full: adds raw diagnostic AoA per axis and figure-of-merit scores
    "full_diag": [
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
    ],
}

def point_coord_table(data: pd.DataFrame):
    """Build a lookup dict: point_name -> {'x_m': ..., 'y_m': ..., 'z_m': ...}."""
    return data.groupby("point")[["x_m", "y_m", "z_m"]].first().to_dict("index")

def gaussian_logpdf_diag(x, mu, sigma):
    """
    Log probability of x under a diagonal Gaussian with mean mu and std sigma.
    'Diagonal' means we assume features are independent (no cross-feature correlations).
    This is a simplification that makes the model computationally tractable.

    Log-space computation avoids numerical underflow when multiplying many small probabilities.
    """
    sigma = np.maximum(sigma, 1e-3)
    z = (x - mu) / sigma
    return -0.5 * np.sum(z * z + np.log(2 * np.pi * sigma * sigma))

def fit_models(train: pd.DataFrame, features: list[str]):
    """
    Fit a diagonal Gaussian for each (point, condition) label in the training data.

    Returns a dict: label -> {'mu', 'sigma', 'n', 'point', 'condition'}

    Sigma regularization prevents degenerate (zero-variance) Gaussians that would
    dominate the posterior inappropriately. We enforce that each feature's sigma
    is at least 15% of the global sigma across all training data.
    """
    models = {}
    global_sigma = train[features].std(ddof=1).replace(0, np.nan).fillna(1.0).to_numpy(float)

    for label, g in train.groupby("point_condition"):
        X = g[features].to_numpy(float)
        mu = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0, ddof=1)

        # Regularize: if sigma is missing or too small, use a fraction of the global spread
        sigma = np.where(np.isfinite(sigma), sigma, global_sigma)
        sigma = np.maximum(sigma, 0.15 * global_sigma)
        sigma = np.maximum(sigma, 1e-3)

        models[label] = {
            "mu": mu,
            "sigma": sigma,
            "n": len(g),
            "point": label.split("__")[0],
            "condition": label.split("__")[1],
        }

    return models

def posterior_over_point_condition(x, models, empirical_prior=False):
    """
    Compute the posterior probability over all (point, condition) labels for one measurement x.

    Steps:
      1. Compute log-likelihood of x under each Gaussian model
      2. (Optionally) add log prior proportional to training set frequency
      3. Subtract max for numerical stability (log-sum-exp trick)
      4. Exponentiate and normalize to get a proper probability distribution

    Returns (labels list, probabilities array).
    """
    labels = list(models.keys())
    total = sum(m["n"] for m in models.values())

    logps = []
    for lab in labels:
        m = models[lab]
        lp = gaussian_logpdf_diag(x, m["mu"], m["sigma"])
        if empirical_prior:
            lp += math.log(m["n"] / total)  # Weight by how often this location was measured
        logps.append(lp)

    logps = np.array(logps)
    logps -= np.max(logps)  # Numerical stability: prevents exp() overflow
    probs = np.exp(logps)
    probs /= probs.sum()

    return labels, probs

def entropy(probs):
    """
    Shannon entropy in bits. High entropy = uncertain prediction (spread over many locations).
    Low entropy = confident prediction (probability concentrated on one location).
    """
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))

def expected_calibration_error(conf, correct, n_bins=10):
    """
    Expected Calibration Error (ECE): measures whether the model's confidence scores
    are accurate. A perfectly calibrated model has accuracy == confidence in every bin.

    Example: if the model says "70% confident" on 100 predictions, it should be right ~70 times.
    If it's right only 50 times, the model is overconfident (ECE will be high).

    Returns the ECE scalar and a DataFrame of per-bin statistics for plotting.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)

    ece = 0.0
    rows = []

    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        if not np.any(mask):
            rows.append({
                "bin": i, "lo": lo, "hi": hi, "n": 0,
                "avg_conf": np.nan, "accuracy": np.nan, "gap": np.nan,
            })
            continue

        avg_conf = float(np.mean(conf[mask]))
        acc = float(np.mean(correct[mask]))
        gap = abs(avg_conf - acc)
        weight = np.mean(mask)  # Fraction of samples in this bin
        ece += weight * gap

        rows.append({
            "bin": i, "lo": lo, "hi": hi,
            "n": int(mask.sum()),
            "avg_conf": avg_conf,
            "accuracy": acc,
            "gap": gap,
        })

    return float(ece), pd.DataFrame(rows)

def credible_set_metrics(point_probs, true_point, coord, level=0.95):
    """
    Compute 95% credible set: the smallest set of candidate locations such that
    their combined probability is at least 95%.

    If the true location is in this set, the prediction is 'covered'.
    A well-calibrated model should have ~95% coverage.
    The credible radius is the max distance from the top-1 candidate to any set member.
    """
    items = sorted(point_probs.items(), key=lambda kv: kv[1], reverse=True)

    cum = 0.0
    included = []
    for point, prob in items:
        included.append(point)
        cum += prob
        if cum >= level:
            break

    covered = true_point in included

    best_point = items[0][0]
    best_xyz = np.array([coord[best_point]["x_m"], coord[best_point]["y_m"], coord[best_point]["z_m"]])

    radii = []
    for p in included:
        xyz = np.array([coord[p]["x_m"], coord[p]["y_m"], coord[p]["z_m"]])
        radii.append(float(np.linalg.norm(xyz - best_xyz)))

    return {
        "credible95_covered": covered,
        "credible95_size": len(included),
        "credible95_radius_m": max(radii) if radii else 0.0,
    }

def evaluate_feature_set(data: pd.DataFrame, feature_name: str, features: list[str]):
    """
    Full evaluation pipeline for one feature set:
      1. Split data 70/30 stratified by (point, condition)
      2. Fit Gaussian models on training set
      3. Run inference on test set
      4. Compute accuracy, error, calibration, and credible set metrics
    """
    features = [f for f in features if f in data.columns]
    use = data[
        ["point", "condition", "anchor", "sequence_n", "x_m", "y_m", "z_m"] + features
    ].copy()

    use = use.replace([np.inf, -np.inf], np.nan)
    use = use.dropna(subset=features).reset_index(drop=True)

    # Create a combined label for stratified splitting
    use["point_condition"] = use["point"] + "__" + use["condition"]

    # Stratified split: ensures each point_condition appears in both train and test
    train, test = train_test_split(
        use,
        test_size=0.30,
        random_state=7,
        stratify=use["point_condition"],
    )

    coord = point_coord_table(use)
    models = fit_models(train, features)

    pred_rows = []

    for _, r in test.iterrows():
        x = r[features].to_numpy(float)
        labels, pc_probs = posterior_over_point_condition(x, models, empirical_prior=False)

        # Marginalize over condition to get the posterior over point alone
        # i.e., sum probabilities for los and obst variants of each point
        point_probs = {}
        cond_probs = {"los": 0.0, "obst": 0.0}

        for lab, prob in zip(labels, pc_probs):
            p = models[lab]["point"]
            c = models[lab]["condition"]
            point_probs[p] = point_probs.get(p, 0.0) + float(prob)
            cond_probs[c] = cond_probs.get(c, 0.0) + float(prob)

        sorted_points = sorted(point_probs.items(), key=lambda kv: kv[1], reverse=True)
        pred_point = sorted_points[0][0]
        pred_point_prob = sorted_points[0][1]

        pred_condition = max(cond_probs.items(), key=lambda kv: kv[1])[0]
        pred_condition_prob = cond_probs[pred_condition]

        top3_points = [p for p, _ in sorted_points[:3]]
        top5_points = [p for p, _ in sorted_points[:5]]

        true_xyz = np.array([r["x_m"], r["y_m"], r["z_m"]])
        pred_xyz = np.array([
            coord[pred_point]["x_m"],
            coord[pred_point]["y_m"],
            coord[pred_point]["z_m"],
        ])

        err = float(np.linalg.norm(pred_xyz - true_xyz))
        cred = credible_set_metrics(point_probs, r["point"], coord, level=0.95)

        pred_rows.append({
            "feature_set": feature_name,
            "true_point": r["point"],
            "true_condition": r["condition"],
            "anchor": r["anchor"],
            "sequence_n": r["sequence_n"],
            "pred_point": pred_point,
            "pred_condition": pred_condition,
            "point_correct": pred_point == r["point"],
            "top3_correct": r["point"] in top3_points,
            "top5_correct": r["point"] in top5_points,
            "condition_correct": pred_condition == r["condition"],
            "pred_point_prob": pred_point_prob,
            "pred_condition_prob": pred_condition_prob,
            "prob_obst": cond_probs.get("obst", 0.0),
            "posterior_entropy_bits": entropy(np.array(list(point_probs.values()))),
            "error_3d_m": err,
            **cred,
        })

    pred = pd.DataFrame(pred_rows)

    # Calibration: does confidence = accuracy across confidence bins?
    point_ece, point_cal = expected_calibration_error(
        pred["pred_point_prob"], pred["point_correct"], n_bins=10
    )
    cond_ece, cond_cal = expected_calibration_error(
        pred["pred_condition_prob"], pred["condition_correct"], n_bins=10
    )

    point_cal["feature_set"] = feature_name
    point_cal["calibration_target"] = "point"
    cond_cal["feature_set"] = feature_name
    cond_cal["calibration_target"] = "condition"

    # Brier score for obstruction classification: lower = better probability estimates
    brier_obst = brier_score_loss(
        (pred["true_condition"] == "obst").astype(int),
        pred["prob_obst"],
    )

    summary = {
        "feature_set": feature_name,
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(features),
        "point_top1_accuracy": float(pred["point_correct"].mean()),
        "point_top3_accuracy": float(pred["top3_correct"].mean()),
        "point_top5_accuracy": float(pred["top5_correct"].mean()),
        "condition_accuracy": float(pred["condition_correct"].mean()),
        "mean_3d_error_m": float(pred["error_3d_m"].mean()),
        "median_3d_error_m": float(pred["error_3d_m"].median()),
        "p75_3d_error_m": float(pred["error_3d_m"].quantile(0.75)),
        "p90_3d_error_m": float(pred["error_3d_m"].quantile(0.90)),
        "mean_point_confidence": float(pred["pred_point_prob"].mean()),
        "mean_condition_confidence": float(pred["pred_condition_prob"].mean()),
        "point_ece": point_ece,
        "condition_ece": cond_ece,
        "condition_brier_obst": float(brier_obst),
        "credible95_coverage": float(pred["credible95_covered"].mean()),
        "mean_credible95_size": float(pred["credible95_size"].mean()),
        "mean_credible95_radius_m": float(pred["credible95_radius_m"].mean()),
    }

    return summary, pred, pd.concat([point_cal, cond_cal], ignore_index=True)


# Evaluate all three feature sets
all_summaries = []
all_predictions = []
all_calibration = []

for name, feats in FEATURE_SETS.items():
    summary, pred, cal = evaluate_feature_set(df, name, feats)
    all_summaries.append(summary)
    all_predictions.append(pred)
    all_calibration.append(cal)

summary_df = pd.DataFrame(all_summaries).sort_values("mean_3d_error_m")
pred_df = pd.concat(all_predictions, ignore_index=True)
cal_df = pd.concat(all_calibration, ignore_index=True)

summary_df.to_csv(PROCESSED / "latent_bayes_uncertainty_results.csv", index=False)
pred_df.to_csv(PROCESSED / "latent_bayes_uncertainty_predictions.csv", index=False)
cal_df.to_csv(PROCESSED / "latent_bayes_calibration_bins.csv", index=False)

print("\n=== Latent Bayesian uncertainty results ===")
print(summary_df.to_string(index=False))

print("\nWrote:")
print("  processed/latent_bayes_uncertainty_results.csv")
print("  processed/latent_bayes_uncertainty_predictions.csv")
print("  processed/latent_bayes_calibration_bins.csv")
