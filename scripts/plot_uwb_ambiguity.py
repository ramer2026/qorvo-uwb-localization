# plot_uwb_ambiguity.py
#
# Shows why the UWB Explorer GUI -- which gives you distance and AoA -- is not
# enough to uniquely locate a tag in a dense 3D grid.
#
# Left panel:  2D overhead view of the 24 grid points. Concentric dashed rings
#              show the mean distance reported by anchorA for each point. Points
#              that share a ring look identical to the chip from range alone.
#
# Right panel: Same map, but each point now has an AoA ray drawn from anchorA.
#              The shaded wedge shows the ±1 std angular spread across repeated
#              measurements. Overlapping wedges = positions the chip cannot
#              distinguish even with angle added.
#
# Reads from raw/ directly.
# Output: plots/uwb_ambiguity.png
#
# Run with: python scripts/plot_uwb_ambiguity.py

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

RAW = Path("raw")
OUT = Path("plots")
OUT.mkdir(exist_ok=True)

FILENAME_RE = re.compile(
    r"^(?P<point>.+)_(?P<condition>los|obst)_(?P<anchor>anchorA|anchorB)"
    r"_(?P<kind>stdout\.txt|diag\.json)$"
)

ANCHOR_A_XY = np.array([0.0, 0.5])  # anchorA at center of grid


# ── Parsing ───────────────────────────────────────────────────────────────────

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
        if s.startswith("status:"):          current["ok"] = "Ok (0x0)" in s
        elif s.startswith("distance:"):      current["distance_cm"] = first_float()
        elif s.startswith("AoA azimuth:"):   current["aoa_azimuth_deg"] = first_float()
        elif s.startswith("AoA elevation:"): current["aoa_elevation_deg"] = first_float()
    if current and "sequence_n" in current:
        rows.append(current)
    return pd.DataFrame(rows)


# ── Load anchorA LOS stdout only ──────────────────────────────────────────────

print("Parsing raw files...")
frames = []
for p in RAW.glob("*_los_anchorA_stdout.txt"):
    m = FILENAME_RE.match(p.name)
    if m:
        frames.append(parse_stdout(p, m.group("point"), "los", "anchorA"))

df = pd.concat(frames, ignore_index=True)
df = df[(df["ok"] == True) & (df["distance_cm"].fillna(0) > 0)].copy()

# Per-point stats
stats = (
    df.groupby("point")
    .agg(
        x_m=("x_m", "first"),
        y_m=("y_m", "first"),
        z_m=("z_m", "first"),
        dist_mean=("distance_cm", "mean"),
        dist_std=("distance_cm", "std"),
        aoa_mean=("aoa_azimuth_deg", "mean"),
        aoa_std=("aoa_azimuth_deg", "std"),
    )
    .reset_index()
)
stats["dist_mean_m"] = stats["dist_mean"] / 100.0
stats["dist_std_m"]  = stats["dist_std"]  / 100.0

# True distance from anchorA in XY plane
stats["true_dist_m"] = np.sqrt(
    (stats["x_m"] - ANCHOR_A_XY[0]) ** 2 +
    (stats["y_m"] - ANCHOR_A_XY[1]) ** 2
)

# AoA ray direction: angle (deg) is measured at the anchor.
# Convert to radians for drawing rays in the XY plane.
stats["aoa_rad"] = np.deg2rad(stats["aoa_mean"])
stats["aoa_std_rad"] = np.deg2rad(stats["aoa_std"].fillna(5.0))

# Color by z-height so depth is visible
z_levels = sorted(stats["z_m"].unique())
z_palette = {z: c for z, c in zip(z_levels, ["#1565C0", "#2E7D32", "#C62828"])}
z_labels  = {z: f"z = {z:+.2f} m" for z in z_levels}
point_colors = stats["z_m"].map(z_palette).tolist()

print(f"  {len(stats)} points loaded")


# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
fig.suptitle(
    "UWB Explorer: What the chip reports vs. where the tag actually is",
    fontsize=14, fontweight="bold", y=1.02,
)

GRID_LIM = (-0.75, 0.75)
GRID_LIM_Y = (-0.15, 1.15)


def draw_base(ax):
    """Shared grid + anchor marker."""
    ax.set_xlim(GRID_LIM)
    ax.set_ylim(GRID_LIM_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.grid(True, alpha=0.2, linestyle="--")
    # AnchorA
    ax.scatter(*ANCHOR_A_XY, marker="*", s=350, color="#FF6F00",
               zorder=6, label="anchorA")
    ax.annotate("anchorA", ANCHOR_A_XY, xytext=(6, 8),
                textcoords="offset points", fontsize=8.5,
                color="#FF6F00", fontweight="bold")
    # True grid points
    for _, row in stats.iterrows():
        ax.scatter(row["x_m"], row["y_m"],
                   color=z_palette[row["z_m"]], s=70,
                   edgecolors="k", linewidths=0.5, zorder=5)


# ── LEFT: range rings ─────────────────────────────────────────────────────────
ax = axes[0]
draw_base(ax)

drawn_radii = set()
for _, row in stats.iterrows():
    r = row["dist_mean_m"]
    r_std = row["dist_std_m"]
    key = round(r, 2)

    # Shaded annulus showing ± std of distance
    theta = np.linspace(0, 2 * np.pi, 360)
    for dr, alpha in [(r_std, 0.08), (0, 0.0)]:
        pass  # drawn below as a fill

    inner = r - r_std
    outer = r + r_std
    theta = np.linspace(0, 2 * np.pi, 360)
    xi = ANCHOR_A_XY[0] + inner * np.cos(theta)
    yi = ANCHOR_A_XY[1] + inner * np.sin(theta)
    xo = ANCHOR_A_XY[0] + outer * np.cos(theta)
    yo = ANCHOR_A_XY[1] + outer * np.sin(theta)
    ax.fill(
        np.concatenate([xo, xi[::-1]]),
        np.concatenate([yo, yi[::-1]]),
        color=z_palette[row["z_m"]], alpha=0.07, zorder=1,
    )

    # Mean distance ring (dashed)
    if key not in drawn_radii:
        xc = ANCHOR_A_XY[0] + r * np.cos(theta)
        yc = ANCHOR_A_XY[1] + r * np.sin(theta)
        ax.plot(xc, yc, "--", color="gray", linewidth=0.8, alpha=0.5, zorder=2)
        drawn_radii.add(key)

ax.set_title(
    "Ranging only\nEach dashed ring = mean measured distance from anchorA",
    fontsize=10,
)

# Annotation callout
ax.annotate(
    "Same ring =\nindistinguishable\nby range alone",
    xy=(0.05, 0.62), xytext=(0.38, 0.85),
    fontsize=8.5, color="#B71C1C",
    arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B71C1C", alpha=0.9),
)

legend_handles = [
    mpatches.Patch(color=z_palette[z], label=z_labels[z]) for z in z_levels
] + [plt.scatter([], [], marker="*", color="#FF6F00", s=120, label="anchorA")]
ax.legend(handles=legend_handles, fontsize=8.5, loc="upper right")


# ── RIGHT: range rings + AoA rays ────────────────────────────────────────────
ax = axes[1]
draw_base(ax)

for _, row in stats.iterrows():
    r     = row["dist_mean_m"]
    angle = row["aoa_rad"]
    a_std = row["aoa_std_rad"]
    color = z_palette[row["z_m"]]

    # Shaded wedge: ± std of AoA at the measured distance
    theta_lo = angle - a_std
    theta_hi = angle + a_std
    wedge_t  = np.linspace(theta_lo, theta_hi, 60)
    wx = np.concatenate([[ANCHOR_A_XY[0]],
                         ANCHOR_A_XY[0] + r * np.cos(wedge_t),
                         [ANCHOR_A_XY[0]]])
    wy = np.concatenate([[ANCHOR_A_XY[1]],
                         ANCHOR_A_XY[1] + r * np.sin(wedge_t),
                         [ANCHOR_A_XY[1]]])
    ax.fill(wx, wy, color=color, alpha=0.18, zorder=1)

    # Mean AoA ray
    xe = ANCHOR_A_XY[0] + r * np.cos(angle)
    ye = ANCHOR_A_XY[1] + r * np.sin(angle)
    ax.plot([ANCHOR_A_XY[0], xe], [ANCHOR_A_XY[1], ye],
            color=color, linewidth=1.0, alpha=0.6, zorder=2)

ax.set_title(
    "Ranging + AoA (what UWB Explorer shows)\nRay = mean AoA, shaded wedge = ±1 std of AoA",
    fontsize=10,
)

ax.annotate(
    "Overlapping wedges =\nstill ambiguous\neven with angle",
    xy=(-0.1, 0.35), xytext=(-0.58, 0.82),
    fontsize=8.5, color="#B71C1C",
    arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B71C1C", alpha=0.9),
)
ax.legend(handles=legend_handles, fontsize=8.5, loc="upper right")

plt.tight_layout()
out_path = OUT / "uwb_ambiguity.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.close()
