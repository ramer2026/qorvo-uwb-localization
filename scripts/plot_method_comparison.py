# plot_method_comparison.py
#
# Compares 3D localization error across four feature sets using the same model
# (ExtraTrees, GroupKFold CV grouped by point) so the only variable is what
# information the model is allowed to use.
#
# Feature sets:
#   1. Range only      -- distance_cm                      (one number per measurement)
#   2. Range + AoA     -- distance + azimuth + elevation   (everything UWB Explorer shows)
#   3. Range + AoA + RSSI -- above + signal strength
#   4. Full fingerprint -- all of the above + peak SNR, RSL, path1 SNR (our model)
#
# Outputs:
#   plots/method_comparison.png  -- bar chart + CDF of per-point errors
#
# Run with: python scripts/plot_method_comparison.py

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

RAW = Path("raw")
OUT = Path("plots")
OUT.mkdir(exist_ok=True)

FILENAME_RE = re.compile(
    r"^(?P<point>.+)_(?P<condition>los|obst)_(?P<anchor>anchorA|anchorB)"
    r"_(?P<kind>stdout\.txt|diag\.json)$"
)


# ── Parsing (same logic as rest of pipeline) ──────────────────────────────────

def parse_coord_token(tok, axis):
    rest = tok[len(axis):]
    sign = -1.0 if rest.startswith("m") else 1.0
    rest = rest[1:] if rest.startswith("m") else rest
    return sign * int(rest) / 100.0


def parse_point_coords(point):
    parts = point.split("_")
    return {
        "z_m": parse_coord_token(next(p for p in parts if p.startswith("z")), "z"),
        "x_m": parse_coord_token(next(p for p in parts if p.startswith("x")), "x"),
        "y_m": parse_coord_token(next(p for p in parts if p.startswith("y")), "y"),
    }


def parse_stdout(path, point, condition, anchor):
    rows, current = [], None
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if s.startswith("sequence n:"):
            if current and "sequence_n" in current:
                rows.append(current)
            current = {"point": point, "condition": condition, "anchor": anchor,
                       **parse_point_coords(point)}
            try:
                current["sequence_n"] = int(s.split(":")[-1].strip())
            except Exception:
                current["sequence_n"] = np.nan
        if current is None:
            continue
        def first_float(s=s):
            m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
            return float(m.group(0)) if m else np.nan
        if s.startswith("status:"):        current["ok"] = "Ok (0x0)" in s
        elif s.startswith("distance:"):    current["distance_cm"] = first_float()
        elif s.startswith("AoA azimuth:"): current["aoa_azimuth_deg"] = first_float()
        elif s.startswith("AoA elevation:"): current["aoa_elevation_deg"] = first_float()
        elif s.startswith("rssi:"):        current["rssi_dbm"] = first_float()
    if current and "sequence_n" in current:
        rows.append(current)
    return pd.DataFrame(rows)


def parse_diag(path, point, condition, anchor):
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception:
        return pd.DataFrame()
    rows = []
    for obj in data:
        row = {"point": point, "condition": condition, "anchor": anchor,
               "sequence_n": obj.get("sequence_n")}
        rsl, path1_rsl, path1_snr, peak_snr = [], [], [], []
        for report in obj.get("reports", []):
            for field in report.get("fields", []):
                for sm in field.get("segment_metrics", []):
                    if "rsl_dbm" in sm:       rsl.append(sm["rsl_dbm"])
                    if "path1_rsl_dbm" in sm: path1_rsl.append(sm["path1_rsl_dbm"])
                    if "path1_snr" in sm:     path1_snr.append(sm["path1_snr"])
                    if "peak_snr" in sm:      peak_snr.append(sm["peak_snr"])
        def mn(v): return float(np.mean(v)) if v else np.nan
        row.update({"diag_rsl_dbm_mean": mn(rsl), "diag_path1_rsl_dbm_mean": mn(path1_rsl),
                    "diag_path1_snr_mean": mn(path1_snr), "diag_peak_snr_mean": mn(peak_snr)})
        rows.append(row)
    return pd.DataFrame(rows)


# ── Load + clean ──────────────────────────────────────────────────────────────

print("Parsing raw files...")
stdout_frames, diag_frames = [], []
for p in RAW.glob("*"):
    m = FILENAME_RE.match(p.name)
    if not m:
        continue
    d = m.groupdict()
    if d["kind"] == "stdout.txt":
        stdout_frames.append(parse_stdout(p, d["point"], d["condition"], d["anchor"]))
    else:
        diag_frames.append(parse_diag(p, d["point"], d["condition"], d["anchor"]))

stdout = pd.concat(stdout_frames, ignore_index=True)
diag   = pd.concat(diag_frames,   ignore_index=True)

df = stdout.merge(diag, on=["point", "condition", "anchor", "sequence_n"], how="left")
df = df[(df["ok"] == True) & (df["distance_cm"].fillna(0) > 0)].copy()
df["anchor_id"] = (df["anchor"] == "anchorB").astype(float)  # 0 = anchorA, 1 = anchorB

print(f"  {len(df)} valid measurements across {df['point'].nunique()} points")

# ── Feature sets to compare ───────────────────────────────────────────────────

FEATURE_SETS = {
    "Range only\n(distance_cm)": [
        "distance_cm", "anchor_id",
    ],
    "Range + AoA\n(UWB Explorer)": [
        "distance_cm", "aoa_azimuth_deg", "aoa_elevation_deg", "anchor_id",
    ],
    "Range + AoA\n+ RSSI": [
        "distance_cm", "aoa_azimuth_deg", "aoa_elevation_deg",
        "rssi_dbm", "anchor_id",
    ],
    "Full fingerprint\n(our model)": [
        "distance_cm", "aoa_azimuth_deg", "aoa_elevation_deg",
        "rssi_dbm", "diag_peak_snr_mean", "diag_rsl_dbm_mean",
        "diag_path1_snr_mean", "anchor_id",
    ],
}

TARGET = ["x_m", "y_m", "z_m"]
COLORS = ["#E53935", "#FB8C00", "#FDD835", "#43A047"]

# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cv(df, features):
    data = df.dropna(subset=features + TARGET).copy()
    X = data[features].replace([np.inf, -np.inf], np.nan).fillna(data[features].median())
    y = data[TARGET].to_numpy(float)
    groups = data["point"].values

    model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    cv = GroupKFold(n_splits=5)
    errors = []
    for train_idx, test_idx in cv.split(X, y, groups):
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        errs = np.linalg.norm(pred - y[test_idx], axis=1)
        errors.extend(errs.tolist())
    return np.array(errors)


results = {}
for label, features in FEATURE_SETS.items():
    print(f"  Running CV: {label.replace(chr(10), ' ')}")
    results[label] = run_cv(df, features)

print("Done. Building plot...")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("3D Localization Error: Ranging + AoA vs Fingerprinting",
             fontsize=14, fontweight="bold", y=1.02)

labels  = list(results.keys())
means   = [np.mean(v)   for v in results.values()]
medians = [np.median(v) for v in results.values()]
p90s    = [np.percentile(v, 90) for v in results.values()]

# ── LEFT: Bar chart of mean / median / P90 ────────────────────────────────────
ax = axes[0]
x = np.arange(len(labels))
w = 0.25

b1 = ax.bar(x - w,   means,   width=w, color=COLORS, alpha=0.9, label="Mean",   edgecolor="k", linewidth=0.6)
b2 = ax.bar(x,       medians, width=w, color=COLORS, alpha=0.6, label="Median", edgecolor="k", linewidth=0.6, hatch="//")
b3 = ax.bar(x + w,   p90s,    width=w, color=COLORS, alpha=0.35, label="P90",   edgecolor="k", linewidth=0.6, hatch="xx")

# Value labels on the mean bars
for bar, val in zip(b1, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.2f} m", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Improvement arrow from "Range + AoA" to "Full fingerprint"
y_arrow = max(means) * 1.18
ax.annotate(
    f"{((means[1] - means[3]) / means[1] * 100):.0f}% lower\nmean error",
    xy=(x[3] - w, means[3]),
    xytext=(x[1] - w, y_arrow),
    fontsize=9, color="#1565C0", fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="-[,widthB=2.5", color="#1565C0", lw=1.5),
)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("3D localization error (m)", fontsize=11)
ax.set_title("Mean / Median / P90 error by feature set\n(ExtraTrees, 5-fold CV grouped by point)", fontsize=10)
ax.set_ylim(0, max(p90s) * 1.35)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0.26, color="#1565C0", linestyle="--", linewidth=1.2, alpha=0.6)
ax.text(len(labels) - 0.5, 0.27, "cross-trial result (0.26 m)", color="#1565C0",
        fontsize=8, ha="right", style="italic")

# ── RIGHT: CDF of per-measurement errors ──────────────────────────────────────
ax = axes[1]
for (label, errors), color in zip(results.items(), COLORS):
    sorted_e = np.sort(errors)
    cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
    ax.plot(sorted_e, cdf, color=color, linewidth=2.2,
            label=label.replace("\n", " "))

# Reference lines
for ref, lbl in [(0.5, "0.5 m"), (1.0, "1.0 m")]:
    ax.axvline(ref, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(ref + 0.01, 0.02, lbl, color="gray", fontsize=8, va="bottom")

ax.set_xlabel("3D localization error (m)", fontsize=11)
ax.set_ylabel("Cumulative fraction of measurements", fontsize=11)
ax.set_title("CDF of per-measurement 3D error\n(further right = worse)", fontsize=10)
ax.set_xlim(left=0)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT / "method_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

# ── Print summary table ───────────────────────────────────────────────────────
print()
print(f"{'Method':<35} {'Mean':>7} {'Median':>8} {'P90':>7}")
print("-" * 60)
for label, errors in results.items():
    name = label.replace("\n", " ")
    print(f"{name:<35} {np.mean(errors):>6.3f}m {np.median(errors):>7.3f}m {np.percentile(errors,90):>6.3f}m")
