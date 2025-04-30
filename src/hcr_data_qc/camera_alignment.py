"""Functions for visualizing camera alignment QC metrics."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from typing import Any, Dict, Tuple


def load_tile_metrics(json_path: Path) -> Dict[str, Any]:
    """Loads tile metrics data from a JSON file.

    Args:
        json_path: Path to the tile_metrics.json file.

    Returns:
        A dictionary containing the loaded tile metrics data.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def _extract_translation_stats(
    metrics_data: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """Extracts translation statistics (dx, dy, d) per pair and combined.

    Args:
        metrics_data: The loaded tile metrics data.

    Returns:
        A tuple containing:
            - stats: Dictionary mapping channel pair to {'dx': array, 'dy': array, 'd': array}.
            - dx_all: Combined numpy array of all dx values.
            - dy_all: Combined numpy array of all dy values.
            - d_all: Combined numpy array of all d (hypotenuse) values.
    """
    stats = {}
    dx_all, dy_all, d_all = [], [], []

    for pair, tiles in metrics_data.items():
        dx, dy = [], []
        for info in tiles.values():
            tx = info["affine_transform"][0][2]
            ty = info["affine_transform"][1][2]
            dx.append(tx)
            dy.append(ty)
        dx = np.array(dx)
        dy = np.array(dy)
        d = np.hypot(dx, dy)
        stats[pair] = {"dx": dx, "dy": dy, "d": d}
        dx_all.extend(dx)
        dy_all.extend(dy)
        d_all.extend(d)

    dx_all, dy_all, d_all = map(np.asarray, (dx_all, dy_all, d_all))
    return stats, dx_all, dy_all, d_all


def plot_translation_distributions(metrics_data: Dict[str, Any]) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the distribution of translations for each channel pair.

    Args:
        metrics_data (dict): A dictionary containing the metrics data.

    Returns:
        fig: The figure object.
        axes: The axes object.

    """
    # ---------------------- 1. Extract translations ----------------------
    stats, dx_all, dy_all, d_all = _extract_translation_stats(metrics_data)

    # ---------------------- 2. Bins  ------------------------
    edges_dx = np.histogram_bin_edges(dx_all, bins="auto")
    edges_dy = np.histogram_bin_edges(dy_all, bins="auto")
    edges_d = np.histogram_bin_edges(d_all, bins="auto")
    edges = {"dx": edges_dx, "dy": edges_dy, "d": edges_d}

    # ---------------------- 3. Shared axis limits ------------------------
    max_abs_xy = max(abs(dx_all).max(), abs(dy_all).max())
    xlim_xy = (-max_abs_xy, max_abs_xy)

    xlim_d = (0, d_all.max())

    # same Y-(count) limit per column for fair comparison
    # ≈ one bin could hold at most N values; safest is count of tiles in biggest pair
    max_count = max(len(v["dx"]) for v in stats.values())
    ylim_count = (0, max_count + 1)

    # ---------------------- 4. Figure layout -----------------------------
    pairs_sorted = sorted(stats.keys())
    n_pairs = len(pairs_sorted)

    fig, axes = plt.subplots(nrows=n_pairs, ncols=3, figsize=(9, 2.6 * n_pairs), sharex="col", sharey="col")

    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    cols = ["dx", "dy", "d"]
    titles = ["X translation (px)", "Y translation (px)", "Shift magnitude |D| (px)"]
    xlims = {"dx": xlim_xy, "dy": xlim_xy, "d": xlim_d}

    for r, pair in enumerate(pairs_sorted):
        for c, key in enumerate(cols):
            ax = axes[r, c]
            vals = stats[pair][key]
            # bins = "auto"
            ax.hist(vals, bins=edges[key], color="tab:blue", edgecolor="black", alpha=0.7)
            ax.set_xlim(xlims[key])
            ax.set_ylim(ylim_count)
            # median line & annotation
            median = np.median(vals)
            ax.axvline(median, ls="--", lw=1, color="red")
            ax.text(
                median, ylim_count[1] * 0.9, f"{median:.2f}", rotation=90, ha="right", va="top", color="red", fontsize=7
            )

            # labels / titles
            if r == 0:
                ax.set_title(titles[c], fontsize=10, pad=6)
            if c == 0:
                ax.set_ylabel(f"{pair}\ncount")
            ax.grid(ls=":", lw=0.4, alpha=0.6)

    fig.suptitle("Distribution of translations", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return fig, axes


def plot_translation_vectors(metrics_data: Dict[str, Any]) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the translation vectors for each channel pair.

    Args:
        metrics_data (dict): A dictionary containing the metrics data.

    Returns:
        fig: The figure object.
        ax: The axes object.

    """

    # ---------------------- 1. Settings ----------------------
    tile_px = 145  # width / height of a tile in *pixels*
    pad_frac = 0.10  # extra white-space around worst vector
    shaft_width = 0.03  # linewidth (in tile-units)
    head_width = 0.035  # arrow-head width   (tile-units)
    head_length = 0.05  # arrow-head length  (tile-units)

    # ---------------------- 2. Collect records and normalize shifts ----------------------
    records, max_shift_f = [], 0.0
    min_x = min_y = math.inf
    max_x = max_y = -math.inf

    for pair, tiles in metrics_data.items():
        for name, info in tiles.items():
            x_idx = int(name.split("_")[2])
            y_idx = int(name.split("_")[4])

            tx_px = info["affine_transform"][0][2]
            ty_px = info["affine_transform"][1][2]

            dx_f, dy_f = tx_px / tile_px, ty_px / tile_px  # +y is "down" in image space
            records.append((pair, x_idx, y_idx, dx_f, dy_f))

            max_shift_f = max(max_shift_f, abs(dx_f), abs(dy_f))
            min_x, max_x = min(min_x, x_idx), max(max_x, x_idx)
            min_y, max_y = min(min_y, y_idx), max(max_y, y_idx)

    R = np.array(records, dtype=[("pair", "U32"), ("x", int), ("y", int), ("dx", float), ("dy", float)])
    pairs = np.unique(R["pair"])
    colors = plt.get_cmap("tab10", len(pairs))

    # ---------------------- 3. Grid geometry & scaling ----------------------
    tile_step = 1 + 2 * (max_shift_f + pad_frac)  # centre–to–centre spacing
    arrow_scale = 0.4 / (max_shift_f + 1e-9)  # stretch so longest ≈ 40 % of half-tile

    # axis limits
    xlim = (-0.5, (max_x + 0.5) * tile_step)
    ylim = (-0.5, (max_y + 0.5) * tile_step)

    # ---------------------- 4. Plot ----------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")

    # 4a. draw grey tile outlines
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            xc, yc = x * tile_step, y * tile_step
            ax.add_patch(Rectangle((xc - 0.5, yc - 0.5), 1, 1, fill=False, ec="lightgray", lw=0.8))

    # 4b. highlight the *top-left* tile with a thicker frame
    top_xc, top_yc = min_x * tile_step, min_y * tile_step
    ax.add_patch(Rectangle((top_xc - 0.5, top_yc - 0.5), 1, 1, fill=False, ec="black", lw=1.5))

    # 4c. little "axis" inside that tile: 1-pixel right & down
    one_px_f = 1.0 / tile_px  # one pixel in tile-units
    axis_len = one_px_f * arrow_scale  # converted to plot units
    ax.arrow(
        top_xc - 0.25,
        top_yc - 0.25,
        axis_len,
        0,
        head_width=head_width,
        head_length=head_length,
        fc="k",
        ec="k",
        lw=shaft_width * 72,
    )
    ax.text(top_xc - 0.25 + axis_len + 0.05, top_yc - 0.25, "1 px", va="center", fontsize=8)
    ax.arrow(
        top_xc - 0.25,
        top_yc - 0.25,
        0,
        -axis_len,
        head_width=head_width,
        head_length=head_length,
        fc="k",
        ec="k",
        lw=shaft_width * 72,
    )

    # 4d. overlay vectors for **all** channel pairs
    for i, pair in enumerate(pairs):
        colour = colors(i)
        subset = R[R["pair"] == pair]
        for rec in subset:
            xc, yc = rec["x"] * tile_step, rec["y"] * tile_step
            dx = rec["dx"] * arrow_scale
            dy = -rec["dy"] * arrow_scale  # invert so +y drifts *down* in image → up in plot
            ax.arrow(
                xc,
                yc,
                dx,
                dy,
                head_width=head_width,
                head_length=head_length,
                length_includes_head=True,
                lw=shaft_width * 72,
                fc=colour,
                ec=colour,
            )

    # ---------------------- 5. Cosmetics ----------------------
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()  # (0,0) still top-left logically

    xtick_pos = [i * tile_step for i in range(min_x, max_x + 1)]
    ytick_pos = [j * tile_step for j in range(min_y, max_y + 1)]
    ax.set_xticks(xtick_pos)
    ax.set_yticks(ytick_pos)
    ax.set_xticklabels(range(min_x, max_x + 1))  # integer tile indices
    ax.set_yticklabels(range(min_y, max_y + 1))
    ax.grid(ls=":", lw=0.4, color="lightgray")

    handles = [plt.Line2D([0], [0], color=colors(i), lw=2, label=p) for i, p in enumerate(pairs)]
    ax.legend(handles=handles, title="Channel pair", bbox_to_anchor=(1.05, 1))

    ax.set_xlabel("tile X index")
    ax.set_ylabel("tile Y index")
    ax.set_title("Overlay of per-tile translation vectors (all channel pairs)")

    plt.tight_layout()
    plt.show()

    return fig, ax
