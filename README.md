# UWB Indoor Localization with Bayesian Fingerprinting
### Qorvo QM35825 | Embedded Systems and Wireless IoT

---

## a. Introduction

### i. Problem Statement and Architecture

The Qorvo QM35825 measures distance to sub-10 cm accuracy and reports angle-of-arrival in real time. The problem is that distance from one anchor is a sphere, and angle narrows it to a cone. Even with two anchors, the intersection of two cones in 3D space is not unique. Raw ranging and AoA alone cannot resolve a 3D position.

This project fixes that with fingerprinting. We collected measurements at 24 fixed 3D grid points arranged in a 1x1 m square, under both line-of-sight and obstructed conditions, across both anchors. Every location builds up a statistical signature across distance, AoA azimuth, AoA elevation, RSSI, and signal diagnostics. The model learns those signatures, including what obstructed signals look like, without needing to solve any geometry.

**Why Bayesian fingerprinting on top of a chip that already ranges well?**
AoA cuts 3D localization error by 8-9% over ranging alone, but only where the angular geometry actually separates nearby points. Where obstruction deflects the signal, the angle reading is corrupted and AoA stops helping. The Bayesian model flags those cases with low confidence instead of silently returning a wrong answer. You get a ranked list of candidate locations and a probability score, so you know when to trust the prediction and when not to.

```
UWB Tag (mobile)  <-- IEEE 802.15.4z -->  Anchor A
                  <-- IEEE 802.15.4z -->  Anchor B
                             |
                       USB / log files
                             v
               Host Python Pipeline (7 steps):
               parse -> analyze -> fingerprint ->
               Bayes -> calibrate -> cross-trial -> demo
```

### ii. Performance Summary

| Metric | Value |
|---|---|
| UWB standard | IEEE 802.15.4z (HRP UWB) |
| Measurement area | 1x1 m square, 24 3D grid points |
| Mean 3D error, range only | ~0.70 m |
| Mean 3D error, range + AoA | ~0.64 m (Random Forest) |
| Mean 3D error, cross-trial | ~0.26 m (ExtraTrees, r1 to r2) |
| AoA contribution | ~8-9% error reduction over range alone |
| Obstruction detection accuracy | ~77% |
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
| anchorB | TBD | TBD | TBD |

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

### i. Data Collection

Raw logs are included in `raw/` and `trials/`. Naming convention:
```
{z}_{x}_{y}_{condition}_{anchor}_stdout.txt
{z}_{x}_{y}_{condition}_{anchor}_diag.json
```
Example: `z050_xm050_y100_los_anchorA_stdout.txt`

To re-collect: run `scripts/collect_anchorA.sh` and `scripts/collect_anchorB.sh` with the tag placed at each grid point.

**Live demo (UWB Explorer GUI):** Plug in both anchors, open UWB Explorer, select both ports, start a session. The GUI shows live distance, azimuth, elevation, and RSSI in real time.

### ii. Run the Pipeline

From the project root, run in order:

```bash
python scripts/process_qorvo_dataset.py --raw-dir raw --out-dir processed
python scripts/analyze_qorvo_complete.py
python scripts/aggregate_fingerprint_model.py
python scripts/latent_bayes_uncertainty.py
python scripts/true_posterior_temperature_sweep.py
python scripts/trial_generalization_analysis.py
python scripts/demo_bayes_r2_table.py
```

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
