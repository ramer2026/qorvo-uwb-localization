# demo_bayes_r2_table.py
#
# Step 7 of the pipeline: offline Bayesian localization demo on r2 test data.
#
# Inputs:  processed/measurements_clean.csv              — r1 training data (all 24 points)
#          processed/trials_r1_r2_measurements_clean.csv — combined r1+r2 data (from trial_generalization_analysis.py)
# Outputs: processed/demo_bayes_r2_table.csv    — per-window predictions with top-3 candidates
#          processed/demo_bayes_r2_summary.csv  — aggregate accuracy and error metrics
#
# What this demo shows:
#   For each r2 test location (4 points × 2 conditions = ~8 windows), we treat all
#   measurements in that window as a batch of observations. The Bayesian model scores each
#   candidate (point, condition) by averaging the Gaussian log-likelihoods across the window,
#   then uses temperature scaling to calibrate confidence before outputting a ranked list.
#
#   This simulates a real deployment scenario: a device at an unknown location produces a
#   stream of UWB measurements, and the system must decide where it is.
#
# Run with: python scripts/demo_bayes_r2_table.py
#       or: python scripts/demo_bayes_r2_table.py --temperature 2.5 --max-rows 100

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

PROCESSED = Path("processed")
TRAIN_CSV = PROCESSED / "measurements_clean.csv"
R2_CSV = PROCESSED / "trials_r1_r2_measurements_clean.csv"

# Full set of features — same as latent_bayes_uncertainty.py 'full_diag'
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


def gaussian_logpdf_diag(x, mu, sigma):
    """
    Log probability of observation vector x under a diagonal Gaussian N(mu, sigma).
    Used to score how well each stored location model explains the current measurement.
    """
    sigma = np.maximum(sigma, 1e-6)
    z = (x - mu) / sigma
    return float(-0.5 * np.sum(z * z + np.log(2 * np.pi * sigma * sigma)))


def apply_temperature(probs, temperature):
    """
    Temperature scaling: raise probabilities to power (1/T) and renormalize.
    Higher temperature = flatter (less confident) distribution.
    Default temperature 2.5 was chosen based on the sweep in true_posterior_temperature_sweep.py.
    """
    p = np.maximum(np.asarray(probs, dtype=float), 1e-300)
    p = p ** (1.0 / temperature)
    return p / p.sum()


def fit_models(train, features):
    """
    Fit one diagonal Gaussian per (point, condition, anchor) triple.

    We fit separate models per anchor (A and B) so that at inference time, we can
    combine evidence from both anchors for each candidate location. This is more
    expressive than fitting a single model per point+condition.

    Returns:
      models: dict mapping (point, condition, anchor) -> {'mu', 'sigma', 'n'}
      coords: dict mapping point_name -> {'x_m', 'y_m', 'z_m'}
    """
    train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    global_sigma = train[features].std(ddof=1).replace(0, np.nan).fillna(1.0).to_numpy(float)

    models = {}
    coords = {}

    for (point, condition, anchor), g in train.groupby(["point", "condition", "anchor"]):
        X = g[features].to_numpy(float)
        mu = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0, ddof=1)
        # Regularize sigma to prevent degenerate near-zero values
        sigma = np.where(np.isfinite(sigma), sigma, global_sigma)
        sigma = np.maximum(sigma, 0.15 * global_sigma)
        sigma = np.maximum(sigma, 1e-3)

        models[(point, condition, anchor)] = {"mu": mu, "sigma": sigma, "n": len(g)}
        coords[point] = {
            "x_m": float(g["x_m"].iloc[0]),
            "y_m": float(g["y_m"].iloc[0]),
            "z_m": float(g["z_m"].iloc[0]),
        }

    return models, coords


def predict_window(window, models, coords, features, temperature, max_rows):
    """
    Given a window of measurements (all from the same true location), produce a
    Bayesian location estimate with top-3 candidates and uncertainty metrics.

    Algorithm:
      1. For each candidate (point, condition), average the log-likelihoods of each
         measurement in the window under the candidate's anchor-specific Gaussian.
      2. Convert averaged log-likelihoods to a probability distribution.
      3. Apply temperature scaling to calibrate confidence.
      4. Marginalize over condition to get per-point probabilities.
      5. Report top-3 candidates, prediction error, and 95% credible set size.
    """
    window = window.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    if len(window) > max_rows:
        window = window.sample(max_rows, random_state=42)

    candidate_pcs = sorted(set((p, c) for (p, c, a) in models.keys()))
    scores = {}

    for point, condition in candidate_pcs:
        lls = []
        for _, row in window.iterrows():
            anchor = row["anchor"]
            key = (point, condition, anchor)
            if key not in models:
                continue
            x = row[features].to_numpy(float)
            m = models[key]
            lls.append(gaussian_logpdf_diag(x, m["mu"], m["sigma"]))

        # Average log-likelihood across all measurements in the window
        if lls:
            scores[f"{point}__{condition}"] = float(np.mean(lls))

    labels = list(scores.keys())
    raw = np.array([scores[label] for label in labels], dtype=float)
    raw -= raw.max()  # Log-sum-exp normalization for numerical stability
    probs = np.exp(raw)
    probs /= probs.sum()
    probs = apply_temperature(probs, temperature)

    # Marginalize: sum LOS and obstructed probabilities together per point
    point_probs = {}
    condition_probs = {"los": 0.0, "obst": 0.0}

    for label, prob in zip(labels, probs):
        point, condition = label.split("__")
        point_probs[point] = point_probs.get(point, 0.0) + float(prob)
        condition_probs[condition] = condition_probs.get(condition, 0.0) + float(prob)

    sorted_points = sorted(point_probs.items(), key=lambda kv: kv[1], reverse=True)
    sorted_conditions = sorted(condition_probs.items(), key=lambda kv: kv[1], reverse=True)

    pred_point = sorted_points[0][0]
    pred_condition = sorted_conditions[0][0]

    true_point = str(window["point"].iloc[0])
    true_condition = str(window["condition"].iloc[0])

    true_xyz = np.array([
        float(window["x_m"].iloc[0]),
        float(window["y_m"].iloc[0]),
        float(window["z_m"].iloc[0]),
    ])
    pred_xyz = np.array([
        coords[pred_point]["x_m"],
        coords[pred_point]["y_m"],
        coords[pred_point]["z_m"],
    ])
    error_m = float(np.linalg.norm(pred_xyz - true_xyz))

    # 95% credible set: smallest set of points covering 95% of the probability mass
    cumulative = 0.0
    credible = []
    for point, prob in sorted_points:
        credible.append(point)
        cumulative += prob
        if cumulative >= 0.95:
            break

    return {
        "true_point": true_point,
        "true_condition": true_condition,
        "pred_point": pred_point,
        "pred_condition": pred_condition,
        "error_m": error_m,
        "p_los": condition_probs.get("los", 0.0),
        "p_obst": condition_probs.get("obst", 0.0),
        "top1": sorted_points[0][0],
        "top1_prob": sorted_points[0][1],
        "top2": sorted_points[1][0] if len(sorted_points) > 1 else "",
        "top2_prob": sorted_points[1][1] if len(sorted_points) > 1 else np.nan,
        "top3": sorted_points[2][0] if len(sorted_points) > 2 else "",
        "top3_prob": sorted_points[2][1] if len(sorted_points) > 2 else np.nan,
        "credible95_size": len(credible),  # How many locations are in the 95% confidence set
        "n_rows_used": len(window),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=2.5,
                        help="Temperature for posterior scaling (>1 = less confident). "
                             "Default 2.5 chosen from temperature sweep analysis.")
    parser.add_argument("--max-rows", type=int, default=999999,
                        help="Max measurements to use per window (default: use all)")
    args = parser.parse_args()

    if not TRAIN_CSV.exists():
        raise SystemExit(f"Missing {TRAIN_CSV}")
    if not R2_CSV.exists():
        raise SystemExit(f"Missing {R2_CSV}. Run trial_generalization_analysis.py first.")

    # Train on the full r1 dataset (all 24 points), test on r2 (4 points)
    train = pd.read_csv(TRAIN_CSV)
    trials = pd.read_csv(R2_CSV)
    r2 = trials[trials["trial"] == "r2"].copy()

    # Only use features present in both datasets
    features = [f for f in FEATURES if f in train.columns and f in r2.columns]
    models, coords = fit_models(train, features)

    # Run inference: one prediction per (point, condition) window in r2
    rows = []
    for (_, _), g in r2.groupby(["point", "condition"]):
        rows.append(predict_window(g, models, coords, features, args.temperature, args.max_rows))

    out = pd.DataFrame(rows).sort_values(["true_point", "true_condition"])
    out_path = PROCESSED / "demo_bayes_r2_table.csv"
    out.to_csv(out_path, index=False)

    # Compute correctness flags for summary
    out["top1_correct"] = out["true_point"] == out["pred_point"]
    out["top3_correct"] = (
        (out["true_point"] == out["top1"])
        | (out["true_point"] == out["top2"])
        | (out["true_point"] == out["top3"])
    )
    out["condition_correct"] = out["true_condition"] == out["pred_condition"]

    summary = pd.DataFrame([{
        "n_windows": len(out),
        "top1_point_accuracy": float(out["top1_correct"].mean()),
        "top3_point_accuracy": float(out["top3_correct"].mean()),
        "condition_accuracy": float(out["condition_correct"].mean()),
        "mean_error_m": float(out["error_m"].mean()),
        "median_error_m": float(out["error_m"].median()),
        "mean_credible95_size": float(out["credible95_size"].mean()),
    }])
    summary_path = PROCESSED / "demo_bayes_r2_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print("=== Bayesian r2 demo table ===")
    print(out.to_string(index=False))
    print()
    print("=== Demo summary ===")
    print(summary.to_string(index=False))
    print()
    print(f"Wrote {out_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
