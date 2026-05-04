# Qorvo UWB Localization Analysis

This package contains the cleaned analysis/demo code and data for the Qorvo QM35825 UWB localization experiment.

## Contents

- raw/: original r1 raw Qorvo measurement logs
- trials/r1/raw/: original trial in trial-folder format
- trials/r2/raw/: repeated-trial subset
- processed/: processed CSV outputs used for analysis and demo
- scripts/: final analysis and demo scripts
- plots/ and plots_deeper/: generated plots
- docs/: result summaries

## Setup

Recommended Python version: Python 3.11.

Commands:

    python -m venv analysis_venv
    source analysis_venv/bin/activate
    pip install -r requirements.txt

## Main analysis scripts

Run from the project root:

    python scripts/process_qorvo_dataset.py --raw-dir raw --out-dir processed
    python scripts/analyze_qorvo_complete.py
    python scripts/aggregate_fingerprint_model.py
    python scripts/latent_bayes_uncertainty.py
    python scripts/true_posterior_temperature_sweep.py
    python scripts/trial_generalization_analysis.py

## Demo

Run the offline Bayesian r2 repeated-trial demo:

    python scripts/demo_bayes_r2_table.py
    column -s, -t < processed/demo_bayes_r2_table.csv
    column -s, -t < processed/demo_bayes_r2_summary.csv

The demo feeds r2 repeated-trial measurements into the Bayesian fingerprint model and outputs predicted location, top candidate points, LOS/obstructed probabilities, and 95% credible-set size.

## Key outputs

- processed/measurements_clean.csv
- processed/los_vs_obst_stats.csv
- processed/aggregate_localization_results.csv
- processed/latent_bayes_uncertainty_results.csv
- processed/true_posterior_temperature_sweep.csv
- processed/trial_generalization_r1_to_r2_results.csv
- processed/demo_bayes_r2_table.csv
- docs/final_results_summary.txt
- docs/repeated_trial_summary.txt

## Note

This package does not include a Python virtual environment. Recreate it using requirements.txt.

The Qorvo firmware was not modified. The main contribution is host-side parsing, analysis, Bayesian localization, calibration, repeated-trial validation, and the offline demo pipeline.
