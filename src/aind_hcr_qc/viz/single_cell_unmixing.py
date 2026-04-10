"""
Single-cell spectral unmixing QC plots.

Three complementary views for one cell × one round:

1. ``plot_spot_projection``  — 3-D spots projected to XY, before/after/removed
   panels plus per-channel barplots.
2. ``plot_spot_measure_distributions`` — violin + strip of per-spot quality
   measures (r, dist, dye_line_dist_ratio) after unmixing.
3. ``plot_cell_qc`` — combined figure with both plots stacked via subfigures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from aind_hcr_qc.utils import saveable_plot

from aind_hcr_qc.constants import Z1_CHANNEL_CMAP_VIBRANT

CHAN_ORDER = ["488", "514", "561", "594", "638"]
CHAN_COLORS = {k: v for k, v in Z1_CHANNEL_CMAP_VIBRANT.items() if k in CHAN_ORDER}


@saveable_plot
def plot_spot_projection(
    m_cell,
    u_cell,
    cell_id,
    round_key,
    chan_order=CHAN_ORDER,
    chan_colors=CHAN_COLORS,
    figsize=(20, 5),
    removed_style="outline",
    subfig=None,
):
    """
    Plot 3-D spots projected to XY before unmixing, after unmixing, and
    a third panel showing only the removed spots coloured by channel.
    A fourth and fifth panel (right) show barplots of n_removed and n_retained
    per channel.

    Parameters
    ----------
    m_cell : pd.DataFrame
        Mixed spots table for the cell/round, with columns ``x``, ``y``,
        ``chan``, ``spot_uid``, and optionally ``removed``.
    u_cell : pd.DataFrame
        Unmixed spots table for the cell/round, with columns ``x``, ``y``,
        ``unmixed_chan``, and optionally ``reassigned``.
    cell_id : int or str
        Cell identifier used in the figure title.
    round_key : str
        Round identifier used in the figure title (e.g. ``"R1"``).
    removed_style : ``"x"`` | ``"outline"``
        How to mark removed spots on the before panel.
    subfig : matplotlib.figure.SubFigure, optional
        If provided, draw into this subfigure instead of creating a new figure.
        The caller is responsible for calling ``plt.show()``.
    """
    has_removed = "removed" in m_cell.columns

    if subfig is None:
        fig = plt.figure(figsize=(figsize[0] + 6, figsize[1]))
        _standalone = True
    else:
        fig = subfig
        _standalone = False

    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 0.45, 0.45], wspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0, sharey=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0, sharey=ax0)
    ax3 = fig.add_subplot(gs[3])
    ax4 = fig.add_subplot(gs[4])
    axes = [ax0, ax1, ax2]

    fig.suptitle(
        f"Cell {cell_id}  Round {round_key} — 3-D spots projected to XY",
        fontsize=12,
    )

    # ── panel 0: before unmixing ─────────────────────────────────────────────
    removed_label_added = False
    for chan in [c for c in chan_order if c in m_cell["chan"].values]:
        sub = m_cell[m_cell["chan"] == chan]
        if has_removed and removed_style == "outline":
            kept = sub[~sub["removed"]]
            rm = sub[sub["removed"]]
            if len(kept):
                axes[0].scatter(
                    kept["x"], kept["y"],
                    c=chan_colors[chan], s=18, alpha=0.65,
                    linewidths=0, label=f"Ch {chan}",
                )
            if len(rm):
                lbl = "removed" if not removed_label_added else "_nolegend_"
                removed_label_added = True
                axes[0].scatter(
                    rm["x"], rm["y"],
                    c=chan_colors[chan], s=22, alpha=0.65,
                    edgecolors="black", linewidths=1.0,
                    zorder=5, label=lbl,
                )
        else:
            axes[0].scatter(
                sub["x"], sub["y"],
                c=chan_colors[chan], s=18, alpha=0.65,
                linewidths=0, label=f"Ch {chan}",
            )

    if has_removed and removed_style == "x":
        rm = m_cell[m_cell["removed"]]
        if len(rm):
            axes[0].scatter(
                rm["x"], rm["y"],
                marker="x", c="black", s=30, linewidths=0.9,
                zorder=5, label="removed",
            )

    # ── panel 1: after unmixing ──────────────────────────────────────────────
    for chan in [c for c in chan_order if c in u_cell["unmixed_chan"].values]:
        sub = u_cell[u_cell["unmixed_chan"] == chan]
        axes[1].scatter(
            sub["x"], sub["y"],
            c=chan_colors[chan], s=18, alpha=0.65,
            linewidths=0, label=f"Ch {chan}",
        )

    ra = u_cell[u_cell["reassigned"]] if "reassigned" in u_cell.columns else u_cell.iloc[:0]
    if len(ra):
        axes[1].scatter(
            ra["x"], ra["y"],
            facecolors="none", edgecolors="black",
            linewidths=0.8, s=40, zorder=5, label="reassigned",
        )

    # ── panel 2: removed spots only ──────────────────────────────────────────
    if has_removed:
        rm_all = m_cell[m_cell["removed"]]
        for chan in [c for c in chan_order if c in rm_all["chan"].values]:
            sub = rm_all[rm_all["chan"] == chan]
            axes[2].scatter(
                sub["x"], sub["y"],
                c=chan_colors[chan], s=22, alpha=0.75,
                edgecolors="black", linewidths=0.8,
                label=f"Ch {chan}",
            )
        if len(rm_all) == 0:
            axes[2].text(
                0.5, 0.5, "no removed spots",
                transform=axes[2].transAxes,
                ha="center", va="center", fontsize=9, color="grey",
            )

    # ── titles, labels, legends ──────────────────────────────────────────────
    titles = [
        "Before unmixing  (detected channel)",
        "After unmixing   (assigned channel)",
        f"Removed spots only  (n={m_cell['removed'].sum() if has_removed else 0})",
    ]
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (pixels)")
        ax.set_aspect("equal")
        if i == 0:
            ax.set_ylabel("y (pixels)")
            ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
        else:
            ax.tick_params(labelleft=False)

    ax0.invert_yaxis()

    # ── panel 3: n_removed per channel barplot ───────────────────────────────
    if has_removed:
        ch_counts = (
            m_cell[m_cell["removed"]]
            .groupby("chan")
            .size()
            .reindex([c for c in chan_order if c in m_cell["chan"].values], fill_value=0)
        )
        bar_colors = [chan_colors.get(ch, "grey") for ch in ch_counts.index]
        ax3.barh(
            ch_counts.index, ch_counts.values,
            color=bar_colors, edgecolor="white", linewidth=0.5, alpha=0.85,
        )
        for y, val in enumerate(ch_counts.values):
            ax3.text(
                val + max(ch_counts.values) * 0.03, y,
                str(int(val)), va="center", ha="left", fontsize=8,
            )
        ax3.set_xlim(0, max(ch_counts.values) * 1.25 or 1)
        ax3.set_xlabel("n removed", fontsize=8)
        ax3.set_title("Removed\nper channel", fontsize=9)
        ax3.spines[["top", "right"]].set_visible(False)
    else:
        ax3.axis("off")

    # ── panel 4: n_retained per channel barplot ──────────────────────────────
    if has_removed:
        ch_retained = (
            m_cell[~m_cell["removed"]]
            .groupby("chan")
            .size()
            .reindex([c for c in chan_order if c in m_cell["chan"].values], fill_value=0)
        )
        bar_colors_r = [chan_colors.get(ch, "grey") for ch in ch_retained.index]
        ax4.barh(
            ch_retained.index, ch_retained.values,
            color=bar_colors_r, edgecolor="white", linewidth=0.5, alpha=0.85,
        )
        max_ret = max(ch_retained.values) if max(ch_retained.values) > 0 else 1
        for y, val in enumerate(ch_retained.values):
            ax4.text(
                val + max_ret * 0.03, y,
                str(int(val)), va="center", ha="left", fontsize=8,
            )
        ax4.set_xlim(0, max_ret * 1.25)
        ax4.set_xlabel("n retained", fontsize=8)
        ax4.set_title("Retained\nper channel", fontsize=9)
        ax4.spines[["top", "right"]].set_visible(False)
    else:
        ax4.axis("off")

    if _standalone:
        plt.tight_layout()
        plt.show()


def plot_spot_measure_distributions(
    unmixed_df,
    measures=("r", "dist", "dye_line_dist_ratio"),
    measure_labels=None,
    chan_order=CHAN_ORDER,
    chan_colors=CHAN_COLORS,
    min_n_violin=5,
    thresholds=None,
    subfig=None,
):
    """
    Horizontal violin + strip plots of per-spot quality measures after unmixing.
    One subplot per measure arranged side-by-side (1 row × N cols).
    Y-axis = unmixed channel, X-axis = measure value.

    Parameters
    ----------
    unmixed_df : pd.DataFrame
        Unmixed spots table with ``unmixed_chan`` and the requested measure
        columns.
    measures : tuple of str
        Measure column names to plot.  Columns absent from ``unmixed_df`` are
        silently skipped.
    thresholds : dict, optional
        Mapping of measure name → threshold value.  Draws a vertical dashed
        line.  Defaults to
        ``{"r": 0.2, "dist": 2, "dye_line_dist_ratio": 1}``.
    subfig : matplotlib.figure.SubFigure, optional
        If provided, draw into this subfigure instead of creating a new figure.
        The caller is responsible for calling ``plt.show()``.
    """
    measures = [m for m in measures if m in unmixed_df.columns]
    if not measures:
        missing = [m for m in ("r", "dist", "dye_line_dist_ratio") if m not in unmixed_df.columns]
        print(f"Columns not found: {missing}. Available: {unmixed_df.columns.tolist()}")
        return

    if measure_labels is None:
        measure_labels = {
            "r": "r  (corr to ideal gaussian)",
            "dist": "dist  (center dist to ideal)",
            "dye_line_dist_ratio": "dye_line_dist_ratio (2nd / 1st)",
        }

    if thresholds is None:
        thresholds = {"r": 0.2, "dist": 2, "dye_line_dist_ratio": 1}

    chans_present = [c for c in chan_order if c in unmixed_df["unmixed_chan"].values]
    n_m = len(measures)
    n_ch = len(chans_present)

    if subfig is None:
        fig, axes = plt.subplots(
            1, n_m,
            figsize=(2.5 * n_m, max(2.5, n_ch * 0.75)),
            squeeze=False,
        )
        _standalone = True
    else:
        fig = subfig
        axes = subfig.subplots(1, n_m, squeeze=False)
        _standalone = False

    fig.suptitle(
        "Per-spot quality measures after unmixing",
        fontsize=11, y=1.02,
    )

    rng = np.random.default_rng(42)

    for col, measure in enumerate(measures):
        ax = axes[0, col]

        for yi, ch in enumerate(chans_present):
            vals = unmixed_df.loc[
                unmixed_df["unmixed_chan"] == ch, measure
            ].dropna().values
            n = len(vals)
            color = chan_colors.get(ch, "grey")

            if n >= min_n_violin:
                parts = ax.violinplot(
                    vals, positions=[yi], widths=0.65,
                    showmedians=True, showextrema=False,
                    vert=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.5)
                    pc.set_edgecolor("none")
                parts["cmedians"].set_color("white")
                parts["cmedians"].set_linewidth(1.8)
                parts["cmedians"].set_zorder(4)

            if n > 0:
                jitter = rng.uniform(-0.18, 0.18, size=n)
                ax.scatter(
                    vals, yi + jitter,
                    c=color, s=18, alpha=0.4, linewidths=0, zorder=3,
                )

        if measure in thresholds:
            ax.axvline(
                thresholds[measure], color="black", lw=1.2, ls="--",
                zorder=5, label=f"t = {thresholds[measure]}", alpha=0.5,
            )

        ax.set_yticks(range(n_ch))
        ax.set_yticklabels([f"Ch {c}" for c in chans_present], fontsize=9)
        ax.set_xlabel(measure_labels.get(measure, measure), fontsize=9)
        ax.set_ylim(-0.6, n_ch - 0.4)
        ax.spines[["top", "right"]].set_visible(False)

        if col > 0:
            ax.tick_params(labelleft=False)

    if _standalone:
        plt.tight_layout()
        plt.show()


def plot_cell_qc(
    m_cell,
    u_cell,
    cell_id,
    round_key,
    measures=("r", "dist", "dye_line_dist_ratio"),
    thresholds=None,
    chan_order=CHAN_ORDER,
    chan_colors=CHAN_COLORS,
    removed_style="outline",
    spots_df=None,
    voxel_size=None,
    nn_pairs=None,
    nn_coords=("x", "y", "z"),
    nn_k=1,
    nn_chan_col="unmixed_chan",
    scatter_pairs=None,
    scatter_axis_limits="equal",
):
    """
    Combined single-cell QC figure.

    Top row    : spot projection (before/after/removed + channel bar charts).
    Bottom row : spot quality measures (violins, left) and optionally NN
                 cross-channel distances (right) in the same row.

    Uses ``fig.subfigures`` so each row has its own independent layout.
    When ``spots_df`` is provided the bottom subfigure is split 1×2
    (measures left, NN right); otherwise the measures fill the full row.

    Parameters
    ----------
    m_cell : pd.DataFrame
        Mixed spots table for the cell/round.
    u_cell : pd.DataFrame
        Unmixed spots table for the cell/round.
    cell_id : int or str
        Cell identifier used in the figure title.
    round_key : str
        Round identifier used in the figure title (e.g. ``"R1"``).
    measures : tuple of str
        Quality measure columns to pass to
        :func:`plot_spot_measure_distributions`.
    thresholds : dict, optional
        Per-measure threshold values for the quality measure plot.
    removed_style : ``"x"`` | ``"outline"``
        How to mark removed spots in the projection panel.
    spots_df : pd.DataFrame, optional
        Full (cell-filtered) unmixed spots table passed to
        :func:`plot_spot_nn_distances`.  When ``None`` the NN row is omitted.
    voxel_size : dict, optional
        Physical voxel size in µm/px, e.g. ``{"z": 1.0, "y": 0.24, "x": 0.24}``.
        Forwarded to :func:`plot_spot_nn_distances`.
    nn_pairs : list of (str, str), optional
        Channel pairs for the NN plot.  Defaults to adjacent pairs.
    nn_coords : tuple of str
        Spatial coordinate columns for the NN distance calculation.
    nn_k : int
        k-th nearest neighbour to use in the NN distance plot.
    nn_chan_col : str
        Column to group spots by for the NN plot.
    scatter_pairs : list of (str, str), optional
        Channel pairs for the adjacent-channel scatter row.  When ``None`` the
        scatter row is omitted.
    scatter_axis_limits : ``"auto"`` | ``"equal"``
        Axis limit mode forwarded to :func:`plot_adjacent_channel_scatter`.
    """
    n_ch = len([c for c in chan_order if c in u_cell["unmixed_chan"].values])
    top_h = 5.0
    bot_h = max(2.5, n_ch * 0.75)
    include_nn = spots_df is not None
    include_scatter = scatter_pairs is not None

    n_scatter_pairs = len(scatter_pairs) if include_scatter else 4
    scatter_h = 8.0 if include_scatter else 0.0

    n_rows_fig = 2 + int(include_scatter)
    height_ratios = [top_h, bot_h] + ([scatter_h] if include_scatter else [])
    total_h = top_h + bot_h + scatter_h + 0.8

    fig = plt.figure(figsize=(26, total_h), constrained_layout=True)
    sfigs = fig.subfigures(n_rows_fig, 1, height_ratios=height_ratios, hspace=0.04)

    plot_spot_projection(
        m_cell, u_cell, cell_id, round_key,
        chan_order=chan_order, chan_colors=chan_colors,
        removed_style=removed_style,
        subfig=sfigs[0],
    )

    if include_nn:
        # split second row: measures on the left, NN distances on the right
        n_m = len(list(measures))
        bot_sfigs = sfigs[1].subfigures(1, 2, width_ratios=[n_m, 2])
        plot_spot_measure_distributions(
            u_cell, measures=list(measures), thresholds=thresholds,
            chan_order=chan_order, chan_colors=chan_colors,
            subfig=bot_sfigs[0],
        )
        plot_spot_nn_distances(
            spots_df, cell_id, round_key,
            chan_col=nn_chan_col,
            chan_order=chan_order, chan_colors=chan_colors,
            pairs=nn_pairs,
            coords=nn_coords,
            voxel_size=voxel_size,
            k=nn_k,
            subfig=bot_sfigs[1],
        )
    else:
        plot_spot_measure_distributions(
            u_cell, measures=list(measures), thresholds=thresholds,
            chan_order=chan_order, chan_colors=chan_colors,
            subfig=sfigs[1],
        )

    if include_scatter:
        plot_adjacent_channel_scatter(
            m_cell, u_cell, cell_id, round_key,
            chan_order=chan_order, chan_colors=chan_colors,
            pairs=scatter_pairs,
            axis_limits=scatter_axis_limits,
            subfig=sfigs[2],
        )

    plt.show()




def plot_spot_nn_distances(spots_df, cell_id, round_key,
                           chan_col="unmixed_chan",
                           chan_order=CHAN_ORDER, chan_colors=CHAN_COLORS,
                           pairs=None,
                           coords=("x", "y", "z"),
                           voxel_size=None,
                           k=1,
                           min_n=3,
                           subfig=None):
    """
    For each defined channel pair (A, B), compute the distance from every spot
    in A to its k-th nearest neighbour among spots in B (and vice versa), then
    plot the distributions as horizontal violins.

    Parameters
    ----------
    pairs : list of (str, str), optional
        Channel pairs to evaluate.  Defaults to all adjacent pairs from
        chan_order plus ("488", "594").
    chan_col : str
        Column to group by — "unmixed_chan" (after) or "chan" (before).
    coords : tuple of str
        Spatial coordinate columns (default 3-D: x, y, z; use ("x","y") for 2-D).
    voxel_size : dict or None
        Physical size in µm/px per axis, e.g. {"z": 1.0, "y": 0.24, "x": 0.24}.
        When provided, distances are in µm; otherwise raw pixels.
    k : int
        Which nearest cross-channel neighbour to use (1 = closest).
    min_n : int
        Skip a direction if the source channel has fewer than this many spots.
    subfig : matplotlib.figure.SubFigure, optional
        If provided, draw into this subfigure instead of creating a new figure.
        The caller is responsible for calling ``plt.show()``.
    """
    from scipy.spatial import KDTree

    coord_cols = [c for c in coords if c in spots_df.columns]

    # ── default pairs: adjacent + 488/594 ────────────────────────────────────
    if pairs is None:
        pairs = [(chan_order[i], chan_order[i + 1]) for i in range(len(chan_order) - 1)]
        #if "488" in chan_order and "594" in chan_order:
        #    pairs.append(("488", "594"))

    # ── apply anisotropy correction ───────────────────────────────────────────
    def _scale(df):
        pts = df[coord_cols].values.astype(float)
        if voxel_size is not None:
            pts = pts * np.array([voxel_size.get(c, 1.0) for c in coord_cols])
        return pts

    dist_unit = "µm" if voxel_size is not None else "px"

    # ── compute cross-channel NN distances for each pair direction ────────────
    # For each pair (A, B) compute A→B and B→A distances separately
    rows = []   # (label, color, distances_array)
    for ch_a, ch_b in pairs:
        for src, tgt in [(ch_a, ch_b), (ch_b, ch_a)]:
            df_src = spots_df[spots_df[chan_col] == src]
            df_tgt = spots_df[spots_df[chan_col] == tgt]
            if len(df_src) < min_n or len(df_tgt) < k:
                continue
            tree = KDTree(_scale(df_tgt))
            dists, _ = tree.query(_scale(df_src), k=k, workers=-1)
            d = dists.ravel() if k == 1 else dists[:, -1]
            rows.append((f"{src} → {tgt}", chan_colors.get(src, "grey"), d))

    if not rows:
        print("No pairs with sufficient spots found.")
        return

    # ── figure ────────────────────────────────────────────────────────────────
    dim_label = "×".join(coord_cols)
    if voxel_size is not None:
        scale_str = "  ".join(f"{c}={voxel_size.get(c,1)}" for c in coord_cols)
        title_suffix = f"({dim_label}, {scale_str} µm/px)"
    else:
        title_suffix = f"({dim_label}, pixels)"

    n_rows = len(rows)
    if subfig is None:
        fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, n_rows * 0.65)))
        _standalone = True
    else:
        fig = subfig
        axes = subfig.subplots(1, 2)
        _standalone = False
    fig.suptitle(
        f"Cell {cell_id}  Round {round_key} — {k}-NN cross-channel distance\n"
        f"{title_suffix}",
        fontsize=11,
    )

    rng = np.random.default_rng(42)

    # ── Panel 0: horizontal violin + strip ───────────────────────────────────
    ax = axes[0]
    for yi, (label, color, vals) in enumerate(rows):
        n = len(vals)
        if n >= 5:
            parts = ax.violinplot(vals, positions=[yi], widths=0.7,
                                  showmedians=True, showextrema=False, vert=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.45); pc.set_edgecolor("none")
            parts["cmedians"].set_color("white")
            parts["cmedians"].set_linewidth(1.8)
            parts["cmedians"].set_zorder(4)
        jitter = rng.uniform(-0.18, 0.18, size=n)
        ax.scatter(vals, yi + jitter, c=color, s=18, alpha=0.4, linewidths=0, zorder=3)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel(f"{k}-NN distance to paired channel  ({dist_unit})", fontsize=9)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlim([0,4])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Source → target  ({k}-NN in the other channel)", fontsize=9)

    # ── Panel 1: overlaid CDFs ────────────────────────────────────────────────
    ax2 = axes[1]
    for label, color, vals in rows:
        sorted_v = np.sort(vals)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax2.plot(sorted_v, cdf, color=color, lw=1.8, label=f"{label}  (n={len(vals)})")

    ax2.set_xlabel(f"{k}-NN distance  ({dist_unit})", fontsize=9)
    ax2.set_ylabel("Cumulative fraction", fontsize=9)
    ax2.set_title("CDF per direction", fontsize=9)
    ax2.legend(fontsize=7, framealpha=0.8)
    ax2.spines[["top", "right"]].set_visible(False)

    if _standalone:
        plt.tight_layout()
        plt.show()

    # ── summary stats ─────────────────────────────────────────────────────────
    print(f"\n{k}-NN cross-channel distance summary  ({title_suffix}):")
    print(f"{'Direction':>14}  {'n':>5}  {'median':>8}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    for label, _, vals in rows:
        print(f"{label:>14}  {len(vals):>5}  {np.median(vals):>8.3f}  {np.mean(vals):>8.3f}  "
              f"{np.std(vals):>8.3f}  {np.min(vals):>8.3f}  {np.max(vals):>8.3f}")




def plot_adjacent_channel_scatter(m_cell, u_cell, cell_id, round_key,
                                   chan_order=CHAN_ORDER, chan_colors=CHAN_COLORS,
                                   pairs=None,
                                   axis_limits="equal",
                                   subfig=None):
    """
    Pairwise scatter of channel intensities, before vs after unmixing.

    Parameters
    ----------
    pairs : list of (str, str), optional
        Channel pairs to plot, e.g. [("488", "514"), ("488", "594")].
        Defaults to all adjacent pairs derived from chan_order.
    axis_limits : "auto" | "equal"
        "auto"  — each subplot uses its own independent axis limits.
        "equal" — all subplots share the same x and y limits (union of all data
                  ranges), making cross-panel comparisons accurate.
    subfig : matplotlib.figure.SubFigure, optional
        If provided, draw into this subfigure instead of creating a new figure.
        The caller is responsible for calling ``plt.show()``.
    """
    if pairs is None:
        pairs = [(chan_order[i], chan_order[i + 1]) for i in range(len(chan_order) - 1)]

    if subfig is None:
        fig = plt.figure(figsize=(4 * len(pairs), 8))
        _standalone = True
    else:
        fig = subfig
        _standalone = False

    axes = fig.subplots(2, len(pairs), sharex=False, sharey=False)
    # ensure axes is always 2-D
    if len(pairs) == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f"Cell {cell_id}  Round {round_key} — Channel intensity scatter",
                 fontsize=12)

    for col_idx, (ch_a, ch_b) in enumerate(pairs):
        col_a = f"chan_{ch_a}_intensity"
        col_b = f"chan_{ch_b}_intensity"
        if col_a not in m_cell.columns or col_b not in m_cell.columns:
            for row_idx in range(2):
                axes[row_idx, col_idx].axis("off")
                axes[row_idx, col_idx].set_title(f"Ch {ch_a} vs Ch {ch_b}\n(data missing)", fontsize=9)
            continue

        for row_idx, (df, chan_col) in enumerate([
            (m_cell, "chan"),
            (u_cell, "unmixed_chan"),
        ]):
            ax = axes[row_idx, col_idx]
            chans_to_show = [c for c in [ch_a, ch_b] if c in df[chan_col].values]
            for chan in chans_to_show:
                sub = df[df[chan_col] == chan]
                ax.scatter(sub[col_a], sub[col_b],
                           c=chan_colors[chan], s=20, alpha=0.5,
                           linewidths=0, label=f"Ch {chan}")
            other = df[~df[chan_col].isin(chans_to_show)]
            if len(other):
                ax.scatter(other[col_a], other[col_b],
                           c="lightgrey", s=8, alpha=0.5, linewidths=0, label="other")

            if row_idx == 0:
                ax.set_title(f"Ch {ch_a} vs Ch {ch_b}", fontsize=9)
            ax.set_xlabel(f"Ch {ch_a} intensity", fontsize=8)
            ax.set_ylabel(f"Ch {ch_b} intensity", fontsize=8)
            ax.legend(loc="upper right", fontsize=7, markerscale=1.5, framealpha=0.6)

    for row_idx, label in enumerate(["Before unmixing", "After unmixing"]):
        axes[row_idx, 0].set_ylabel(f"{label}\nCh {pairs[0][1]} intensity", fontsize=8)

    # ── shared axis limits across all panels ─────────────────────────────────
    if axis_limits == "equal":
        all_x, all_y = [], []
        for ax in axes.flat:
            for line in ax.collections:
                offsets = line.get_offsets()
                if len(offsets):
                    all_x.extend(offsets[:, 0])
                    all_y.extend(offsets[:, 1])
        if all_x and all_y:
            pad = 0.05
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            xpad = (xmax - xmin) * pad
            ypad = (ymax - ymin) * pad
            for ax in axes.flat:
                ax.set_xlim(xmin - xpad, xmax + xpad)
                ax.set_ylim(ymin - ypad, ymax + ypad)

    if _standalone:
        plt.tight_layout()
        plt.show()



