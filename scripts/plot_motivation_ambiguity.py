# plot_motivation_ambiguity.py
#
# Motivation plot: why ranging + AoA alone can't resolve 3D position.
#
# Left panel:  "What UWB Explorer shows" -- distance vs AoA azimuth from anchorA (LOS).
#              Points that overlap here are indistinguishable from the GUI alone.
# Right panel: "What fingerprinting sees" -- PCA of the full feature set (distance,
#              AoA azimuth, AoA elevation, RSSI, peak SNR, RSL, path1 SNR).
#              Same 24 locations, separated much more cleanly.
#
# Reads directly from raw/ -- no processed CSVs needed.
# Output: plots/motivation_ambiguity.png
#
# Run with: python scripts/plot_motivation_ambiguity.py

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RAW = Path("raw")
OUT = Path("plots")
OUT.mkdir(exist_ok=True)

FILENAME_RE = re.compile(
    r"^(?P<point>.+)_(?P<condition>los|obst)_(?P<anchor>anchorA|anchorB)"
    r"_(?P<kind>stdout\.txt|diag\.json)$"
)

FEATURES = [
    "distance_cm",
    "aoa_azimuth_deg",
    "aoa_elevation_deg",
    "rssi_dbm",
    "diag_peak_snr_mean",
    "diag_rsl_dbm_mean",
    "diag_path1_snr_mean",
]


# ── Parsing ──────────────────────────────────────────────────────────────────

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
            current = {"point": point, "condition": condition, "anchor": anchor}
            try:
                current["sequence_n"] = int(s.split(":")[-1].strip())
            except Exception:
                current["sequence_n"] = np.nan
        if current is None:
            continue
        def first_float(s=s):
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
                    if "rsl_dbm" in sm:      rsl.append(sm["rsl_dbm"])
                    if "path1_rsl_dbm" in sm: path1_rsl.append(sm["path1_rsl_dbm"])
                    if "path1_snr" in sm:     path1_snr.append(sm["path1_snr"])
                    if "peak_snr" in sm:      peak_snr.append(sm["peak_snr"])
        def mn(v): return float(np.mean(v)) if v else np.nan
        row.update({
            "diag_rsl_dbm_mean":       mn(rsl),
            "diag_path1_rsl_dbm_mean": mn(path1_rsl),
            "diag_path1_snr_mean":     mn(path1_snr),
            "diag_peak_snr_mean":      mn(peak_snr),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ── Load raw data ─────────────────────────────────────────────────────────────

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

merged = stdout.merge(diag, on=["point", "condition", "anchor", "sequence_n"], how="left")
merged = merged[(merged["ok"] == True) & (merged["distance_cm"].fillna(0) > 0)].copy()

# Per-(point, condition, anchor) mean features
summary = (
    merged.groupby(["point", "condition", "anchor"])[FEATURES]
    .mean()
    .reset_index()
)
for col, fn in [("x_m", "x_m"), ("y_m", "y_m"), ("z_m", "z_m")]:
    summary[col] = summary["point"].apply(lambda p, fn=fn: parse_point_coords(p)[fn])

# LOS + anchorA only for this comparison (clean signal, one anchor -- what the GUI shows)
df = summary[(summary["condition"] == "los") & (summary["anchor"] == "anchorA")].copy()
df = df.dropna(subset=FEATURES).reset_index(drop=True)
print(f"  {len(df)} locations after filtering")


# ── Color scheme: by z-height level ──────────────────────────────────────────
z_levels = sorted(df["z_m"].unique())
z_palette = {z: c for z, c in zip(z_levels, ["#2196F3", "#4CAF50", "#F44336"])}
point_colors = [z_palette[z] for z in df["z_m"]]

legend_handles = [
    mpatches.Patch(color=z_palette[z], label=f"z = {z:+.2f} m")
    for z in z_levels
]


# ── PCA on full fingerprint feature set ──────────────────────────────────────
X = df[FEATURES].to_numpy(float)
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_


# ── Find most ambiguous pairs in ranging+AoA space for annotation ────────────
X_gui = df[["distance_cm", "aoa_azimuth_deg"]].to_numpy(float)
X_gui_norm = StandardScaler().fit_transform(X_gui)
D = cdist(X_gui_norm, X_gui_norm)
np.fill_diagonal(D, np.inf)

ambiguous_pairs = []
seen = set()
for i in range(len(X_gui_norm)):
    j = int(np.argmin(D[i]))
    key = tuple(sorted((i, j)))
    if key not in seen:
        seen.add(key)
        ambiguous_pairs.append((i, j, D[i, j]))

ambiguous_pairs.sort(key=lambda t: t[2])
top_pairs = ambiguous_pairs[:4]  # annotate the 4 most ambiguous pairs


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle(
    "Why ranging + AoA alone cannot resolve 3D position",
    fontsize=15, fontweight="bold", y=1.02,
)

# ── LEFT: UWB Explorer space ──────────────────────────────────────────────────
ax = axes[0]
ax.scatter(
    df["distance_cm"], df["aoa_azimuth_deg"],
    c=point_colors, s=130, edgecolors="k", linewidths=0.6, zorder=3,
)

# Draw double-headed arrows between the most ambiguous pairs
for i, j, _ in top_pairs:
    xi, yi = X_gui[i]
    xj, yj = X_gui[j]
    ax.annotate(
        "", xy=(xj, yj), xytext=(xi, yi),
        arrowprops=dict(arrowstyle="<->", color="#CC0000", lw=1.8),
        zorder=4,
    )
    mx, my = (xi + xj) / 2, (yi + yj) / 2
    ax.text(mx + 0.8, my + 0.6, "same to\nthe chip",
            color="#CC0000", fontsize=7.5, fontstyle="italic",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CC0000", alpha=0.85))

ax.set_xlabel("Mean distance from anchorA (cm)", fontsize=11)
ax.set_ylabel("Mean AoA azimuth from anchorA (deg)", fontsize=11)
ax.set_title(
    "UWB Explorer view\n(ranging + AoA from one anchor, LOS)",
    fontsize=11,
)
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_handles, title="Height (z)", fontsize=9,
          title_fontsize=9, loc="upper left")

# ── RIGHT: Fingerprint PCA ────────────────────────────────────────────────────
ax = axes[1]
ax.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=point_colors, s=130, edgecolors="k", linewidths=0.6, zorder=3,
)
ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% variance explained)", fontsize=11)
ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% variance explained)", fontsize=11)
ax.set_title(
    "Fingerprint feature space\n(+ RSSI, peak SNR, RSL, path1 SNR — PCA projection)",
    fontsize=11,
)
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_handles, title="Height (z)", fontsize=9,
          title_fontsize=9, loc="upper left")

feature_note = "Fingerprint features: " + ", ".join(FEATURES)
fig.text(0.5, -0.03, feature_note, ha="center", fontsize=8, color="gray")

plt.tight_layout()
out_path = OUT / "motivation_ambiguity.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.close()
