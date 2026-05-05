# UWB Indoor Localization with Bayesian Fingerprinting
### Qorvo QM35825 | Embedded Systems and Wireless IoT

---

## a. Introduction

### i. Problem Statement and Architecture

The Qorvo QM35825 measures distance to sub-10 cm accuracy and reports angle-of-arrival in real time. The problem is that distance from one anchor is a sphere, and angle narrows it to a cone. Even with two anchors, the intersection of two cones in 3D space is not unique. Raw ranging and AoA alone cannot resolve a 3D position.

This project fixes that with fingerprinting. We collected measurements at 24 fixed 3D grid points arranged in a 1x1 m square, under both line-of-sight and obstructed conditions, across both anchors. Every location builds up a statistical signature across distance, AoA azimuth, AoA elevation, RSSI, and signal diagnostics. The model learns those signatures, including what obstructed signals look like, without needing to solve any geometry.

**Why Bayesian fingerprinting on top of a chip that already ranges well?**
AoA cuts 3D localization error by 8-9% over ranging alone, but only where the angular geometry actually separates nearby points. Where obstruction deflects the signal, the angle reading is corrupted and AoA stops helping. The Bayesian model flags those cases with low confidence instead of silently returning a wrong answer. You get a ranked list of candidate locations and a probability score, so you know when to trust the prediction and when not to.

### ii. Performance Summary

| Metric | Value |
|---|---|
| UWB standard | IEEE 802.15.4z (HRP UWB) |
| Measurement area | 1x1 m square, 24 3D grid points |
| Mean 3D error, range only | ~0.70 m |
| Mean 3D error, range + AoA | ~0.64 m (Random Forest) |
| Mean 3D error, cross-trial | ~0.26 m (ExtraTrees, r1 to r2) |
| AoA contribution | ~8-9% error reduction over range alone |
| Obstruction detection accuracy | ~77% (Random Forest, `analyze_qorvo_complete.py`, GroupKFold CV) |
| Live demo | UWB Explorer GUI |

---

## b. Hardware Details

### i. Development Kit

[Qorvo QM35825DK-05](https://www.qorvo.com/products/p/QM35825DK-05#related-products)

| Component | Details |
|---|---|
| UWB SoC | Qorvo QM35825 |
| Host processor | Raspberry Pi 4 (Broadcom BCM2711, quad-core ARM Cortex-A72 @ 1.8 GHz, 64-bit) |
| RAM | 8 GB LPDDR4X-3200 |
| Antenna | Jolie Quad Qorvo antenna (PDoA-capable, with stand) |
| Interface | USB Type-C |
| Ranging accuracy | ±5 cm |
| AoA accuracy | ±2 degrees (PDoA) |
| UWB frequency bands | 6.5 GHz (Ch. 5) / 8.0 GHz (Ch. 9) |
| UWB bandwidth | ~500 MHz (HRP UWB) |
| Ranging method | Two-Way Ranging (TWR) |
| AoA method | Phase Difference of Arrival (PDoA) |
| Ranging rate | ~10 Hz per anchor |
| Anchor count | 2 (fixed) |
| Tag count | 1 (mobile) |

### ii. Experimental Setup

AnchorA was placed at the center of the 1x1 m measurement grid. The tag was moved by hand to each of the 24 surrounding grid points for both LOS and obstructed conditions.

| Anchor | x (m) | y (m) | z (m) |
|---|---|---|---|
| anchorA | 0.0 | 0.5 | 0.0 |
| anchorB | 0.0 | 0.0 | 0.0 |

Line-of-sight:
![Clear LOS setup](docs/Setup_Clear_LineOfSight.jpeg)

Obstructed:
![Obstructed setup](docs/Setup_Obstructed.jpeg)

---

## c. Software Environment

### i. Firmware

Stock Qorvo QM35825 DK firmware, no modifications. Data collection uses the `run_fira_twr` CLI from the Qorvo UWB SDK (`--en-rssi --en-diag --stats --diag_dump`). All analysis and ML runs on the host PC in Python.

### ii. Host Software

**Dependencies (`requirements.txt`):** numpy, pandas, scipy, scikit-learn, matplotlib

```bash
python3.11 -m venv analysis_venv
source analysis_venv/bin/activate    # Windows: analysis_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

| Tool | Purpose |
|---|---|
| UWB Explorer GUI | Live ranging and AoA visualization |
| `run_fira_twr` CLI | Data collection (called by `collect_anchorA/B.sh`) |
| Python 3.11 | Analysis pipeline |
| VS Code | IDE |

**Host machine (analysis PC):**

| Component | Details |
|---|---|
| Machine | ASUS Vivobook K5504VA |
| OS | Manjaro Linux (Arch), kernel 7.0.3-1-MANJARO, x86_64 |
| CPU | Intel Core i9-13900H (13th Gen Raptor Lake), 14 cores / 20 threads, up to 5.4 GHz boost |
| Cache | L1: 1.2 MB, L2: 11.5 MB, L3: 24 MB |
| Compiler | GCC 15.2.1 |

---

## d. Reproducibility Guide

**Tutorial (UWB Explorer GUI):** Plug in both anchors, open UWB Explorer, select both ports, start a session. The GUI shows live distance, azimuth, elevation, and RSSI in real time.

### i. Data Collection

Raw logs are included in `raw/` and `trials/`. Naming convention:
```
{z}_{x}_{y}_{condition}_{anchor}_stdout.txt
{z}_{x}_{y}_{condition}_{anchor}_diag.json
```
Example: `z050_xm050_y100_los_anchorA_stdout.txt`

Run `scripts/collect_anchorA.sh` and `scripts/collect_anchorB.sh` with the tag placed at each grid point.

To start collecting data, run the following command on each UWB. Start off with the coordinate, indicate if it is obstructed or clear line of sight, then indicate time length in seconds.
```
Example 1: z000_y000_x000 los 60
Example 2: z000_y000_x000 obst 60
```

### ii. Run the Pipeline

Run all scripts from the project root in order. Each step outputs CSVs into `processed/` that the next step reads.

**Step 1 -- Parse raw logs**
```bash
python scripts/process_qorvo_dataset.py --raw-dir raw --out-dir processed
```
Reads all `stdout.txt` and `diag.json` files from `raw/`, merges them by sequence number, computes true range from anchor geometry, and runs LOS vs obstructed statistical tests (Cohen's d, Welch's t-test). Outputs `processed/measurements_clean.csv`.

**Step 2 -- Full analysis and ML models**
```bash
python scripts/analyze_qorvo_complete.py
```
Pivots per-anchor measurements into one row per (point, condition), trains three Random Forest localization models (range only / range+AoA / range+AoA+diagnostics) with GroupKFold CV, trains the obstruction classifier (~77% accuracy), and generates all box/delta plots into `plots/`.

**Step 3 -- Aggregate fingerprint model**
```bash
python scripts/aggregate_fingerprint_model.py
```
Computes 6 statistics per feature per anchor (mean, std, median, IQR, Q10, Q90), trains ExtraTrees regression and Random Forest classifier on the fingerprint table. Tests three feature sets to isolate the AoA contribution.

**Step 4 -- Bayesian uncertainty quantification**
```bash
python scripts/latent_bayes_uncertainty.py
```
Fits one diagonal Gaussian per (point, condition) label, scores each candidate via log-likelihood, marginalizes over condition, and reports accuracy, mean 3D error, Expected Calibration Error (ECE), 95% credible set size, and Brier score.

**Step 5 -- Temperature calibration sweep**
```bash
python scripts/true_posterior_temperature_sweep.py
```
Sweeps temperature T from 1.0 to 50.0, applies `probs**(1/T)` scaling to the Bayesian posterior at each step, and records accuracy and calibration. Used to select T=2.5 as the operating point.

**Step 6 -- Cross-trial generalization**
```bash
python scripts/trial_generalization_analysis.py
```
Trains ExtraTrees on the full r1 dataset (24 points), tests on r2 (4 points, independently collected on a different day). Validates that the fingerprint model generalizes across sessions. Mean 3D error: ~0.26 m.

**Step 7 -- Offline Bayesian demo on r2 test data**
```bash
python scripts/demo_bayes_r2_table.py
```
Runs the Bayesian localization demo on r2: fits per-(point, condition, anchor) Gaussians on r1 training data, averages log-likelihoods over each r2 measurement window, applies temperature scaling (T=2.5), and outputs a ranked top-3 candidate table with probabilities and 95% credible set size.

### iii. Key Output Files

| Metric | Script | Output |
|---|---|---|
| LOS vs obstructed stats | `analyze_qorvo_complete.py` | `processed/los_vs_obst_stats.csv` |
| RF localization error | `analyze_qorvo_complete.py` | `processed/localization_model_results.csv` |
| Fingerprint model error | `aggregate_fingerprint_model.py` | `processed/aggregate_localization_results.csv` |
| Bayesian calibration | `latent_bayes_uncertainty.py` | `processed/latent_bayes_uncertainty_results.csv` |
| Temperature sweep | `true_posterior_temperature_sweep.py` | `processed/true_posterior_temperature_sweep.csv` |
| Cross-trial generalization | `trial_generalization_analysis.py` | `processed/trial_generalization_r1_to_r2_results.csv` |
| Demo predictions | `demo_bayes_r2_table.py` | `processed/demo_bayes_r2_table.csv` |

### iv. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the venv first |
| `FileNotFoundError: measurements_clean.csv` | Run step 1 first |
| `FileNotFoundError: trials_r1_r2_measurements_clean.csv` | Run step 6 before step 7 |
| `Missing folder: trials/r1/raw` | Run all scripts from the project root |

---

## Limitations

**Data collection scripts are machine-specific and require modification to reproduce.**

`collect_anchorA.sh` and `collect_anchorB.sh` were written for the specific machine and Linux distribution used during data collection (Manjaro Linux). Two things must be updated before the scripts will work on a different machine:

**1. Hardcoded paths (both scripts)**

The following paths are hardcoded to the original machine's directory layout and pyenv setup:

```bash
BASE="$HOME/Desktop/qorvo_data_redo"
SDK="$HOME/Desktop/qm35-sdk/Samples/Python/UWB-Qorvo-Tools"
RUN_FIRA_TWR="$HOME/.pyenv/versions/3.11.9/bin/run_fira_twr"
```

Update `BASE` to your desired output directory, `SDK` to where you installed the Qorvo UWB SDK, and `RUN_FIRA_TWR` to the actual path of your `run_fira_twr` binary (find it with `which run_fira_twr` after activating the SDK venv).

**2. USB device detection for anchorB (`collect_anchorB.sh`)**

The script detects the anchorB serial port using pyserial's hwid string:

```python
matches = [p.device for p in ports if "2DE0:1337" in p.hwid or "Raspberry Pi Composite Gadget" in p.description]
```

The hwid string format varies by Linux distro -- on some systems it is uppercase (`2DE0:1337`), on others lowercase (`2de0:1337`). If the script fails to find the device, check what your system reports:

```bash
python -c "import serial.tools.list_ports; [print(p.device, p.hwid, p.description) for p in serial.tools.list_ports.comports()]"
```

Then update the hwid string in the script to match.

The analysis pipeline (Steps 1-7) has no such dependencies and runs on any machine with Python 3.11 and the required packages.

---

## e. Repository Structure

```
.
    raw/                              # r1 raw measurement logs (96 files)
        {z}_{x}_{y}_{cond}_{anchor}_{kind}.{ext}
    trials/
        r1/raw/                       # r1 data in trial-folder format
        r2/raw/                       # r2 repeated-trial subset (4 points)
    scripts/
        process_qorvo_dataset.py        # Step 1: parse raw logs
        analyze_qorvo_complete.py       # Step 2: analysis + ML models
        aggregate_fingerprint_model.py  # Step 3: fingerprint localization, isolates AoA contribution
        latent_bayes_uncertainty.py     # Step 4: Bayesian inference
        true_posterior_temperature_sweep.py  # Step 5: calibration
        trial_generalization_analysis.py     # Step 6: cross-trial test
        demo_bayes_r2_table.py          # Step 7: offline demo
    docs/                             # Setup photos and result summaries
    processed/                        # Generated CSVs (gitignored, recreate via scripts)
    plots/                            # Generated plots (gitignored)
    plots_deeper/                     # Generated plots (gitignored)
    requirements.txt
    .gitignore
    README.md
    intro.md
```

---

## References

- [Qorvo QM35825DK-05 product page](https://www.qorvo.com/products/p/QM35825DK-05#related-products)
- [IEEE 802.15.4z standard](https://standards.ieee.org/ieee/802.15.4z/10375/)
- [Scikit-learn documentation](https://scikit-learn.org)
