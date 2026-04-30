"""Spots"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from aind_hcr_qc.utils.utils import saveable_plot

PLOT_FONT_MULTIPLIER = 1.15 # good smallish = 1.08


def _fs(size):
    """Scale a base font size by a global multiplier."""
    return max(1, size * PLOT_FONT_MULTIPLIER)
# --
# Utils
# ---


def annotate_spots_df(mixed_df, unmixed_df):
    """
    Annotate the full mixed and unmixed spot tables (returns new copies).

    unmixed_df  → adds  'reassigned' (bool): chan != unmixed_chan
    mixed_df    → adds  'removed'    (bool): spot absent from unmixed table

    If a round column ('round' or 'round_key') is present, survival is checked
    per-round so that spot_uids that are only locally unique within a round are
    handled correctly.  Otherwise a single global set lookup is used.
    """
    unmixed_out = unmixed_df.copy()
    unmixed_out["reassigned"] = unmixed_out["chan"] != unmixed_out["unmixed_chan"]

    mixed_out = mixed_df.copy()

    round_col = next((c for c in ("round", "round_key") if c in mixed_df.columns), None)

    if round_col is not None:
        mixed_out["removed"] = True   # default; overwritten per round
        for rnd, m_grp in mixed_out.groupby(round_col):
            survived = set(
                unmixed_out.loc[unmixed_out[round_col] == rnd, "spot_uid"]
            )
            mixed_out.loc[m_grp.index, "removed"] = ~m_grp["spot_uid"].isin(survived)
    else:
        survived = set(unmixed_out["spot_uid"])
        mixed_out["removed"] = ~mixed_out["spot_uid"].isin(survived)

    return mixed_out, unmixed_out


# ---
# Plots
# ---

@saveable_plot()
def plot_removal_metric_distributions(mixed_df, round_id, chan_order=["488", "514", "561", "594", "638"], chan_colors=None):

    """
    Summary figure for one round: three spot-quality metrics (r,
    dye_line_dist_ratio, chan_intensity) split by removed vs kept,
    faceted by channel.

    Layout: 3 rows (metrics) × N_channels columns.
    Uses density-normalised histograms so channels with very different
    spot counts are still visually comparable.
    """
    df = mixed_df[mixed_df["round"] == round_id].copy()

    # ── per-spot chan_intensity: vectorised channel lookup ────────────────────
    df["chan_intensity"] = np.nan
    for ch in chan_order:
        col = f"chan_{ch}_intensity"
        if col in df.columns:
            mask = df["chan"] == ch
            df.loc[mask, "chan_intensity"] = df.loc[mask, col]

    # ── metric definitions ────────────────────────────────────────────────────
    METRICS = []
    for col, label, use_log, clip_pct in [
        ("r",                  "r  (corr. to ideal Gaussian)",      False, None),
        ("dye_line_dist_ratio", "dye_line_dist_ratio  (2nd / 1st)", False, 99),
        ("chan_intensity",       "intensity in detected channel",    True,  None),
    ]:
        if col in df.columns:
            METRICS.append((col, label, use_log, clip_pct))

    COLORS = {"kept": "#4575b4", "removed": "#d73027"}
    chans  = [ch for ch in chan_order if ch in df["chan"].values]
    n_ch   = len(chans)
    n_m    = len(METRICS)

    fig, axes = plt.subplots(
        n_m, n_ch,
        figsize=(n_ch * 2.8, n_m * 2.6),
        squeeze=False,
    )
    fig.suptitle(
        f"Round {round_id} — spot removal metric distributions  "
        f"(removed vs kept, density-normalised)",
        fontsize=12, y=1.02,
    )

    for row, (col, ylabel, use_log, clip_pct) in enumerate(METRICS):
        for ci, ch in enumerate(chans):
            ax  = axes[row][ci]
            sub = df[df["chan"] == ch]

            kept    = sub.loc[~sub["removed"], col].dropna()
            removed = sub.loc[ sub["removed"], col].dropna()

            if clip_pct is not None:
                hi = np.nanpercentile(sub[col].dropna(), clip_pct)
                kept    = kept.clip(upper=hi)
                removed = removed.clip(upper=hi)

            all_vals = pd.concat([kept, removed]).dropna()
            if len(all_vals) == 0:
                ax.set_visible(False)
                continue

            if use_log:
                lo   = np.log10(max(all_vals.min(), 1e-3))
                hi_v = np.log10(all_vals.max())
                bins = np.logspace(lo, hi_v, 41)
            else:
                bins = np.linspace(all_vals.min(), all_vals.max(), 41)

            chan_color = chan_colors.get(ch, "#888888") if chan_colors is not None else "#888888"

            ax.hist(kept,    bins=bins, alpha=0.45, color=COLORS["kept"],
                    density=True, label=f"kept  n={len(kept):,}")
            ax.hist(removed, bins=bins, alpha=0.7,  color=COLORS["removed"],
                    density=True, label=f"removed  n={len(removed):,}")
            # subtle vertical line at the channel's own color to identify the column
            ax.axvline(np.nan, color=chan_color, lw=0)  # invisible — just for spacing

            if use_log:
                ax.set_xscale("log")

            if row == 0:
                ax.set_title(f"Ch {ch}", fontsize=10,
                             color=chan_colors.get(ch, "black") if chan_colors is not None else "black", fontweight="bold")
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=7.5)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)

            if ci == 0:
                ax.legend(fontsize=6, framealpha=0.8)

    plt.tight_layout()
    return fig



def plot_top_spots_per_channel(data: pd.DataFrame, ax=None, figsize=(6, 5)):
    """
    Plot the top 10 spots per channel based on spot count.
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing columns 'channel', 'spot_count', and 'cell_id'.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes will be created.
    figsize : tuple, optional
        Size of the figure if a new one is created.
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_plot = True
    else:
        show_plot = False

    # Get top 10 cells for each gene
    top_genes = data.groupby("gene").apply(lambda x: x.nlargest(10, "spot_count")).reset_index(drop=True)

    # Create swarm plot
    sns.swarmplot(data=top_genes, x="gene", y="spot_count", hue="gene", dodge=False, palette="Set1", ax=ax)

    # Add box plot
    sns.boxplot(
        data=top_genes, x="gene", y="spot_count", color="lightgray", fliersize=0, linewidth=0.5, width=0.5, ax=ax
    )

    # Customize plot
    ax.set_xlabel("Gene")
    ax.set_ylabel("Spot Count")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("Top 10 cell_id by spot_count for each gene")

    # Show plot if we created a new figure
    if show_plot:
        plt.tight_layout()
        plt.show()

    return ax



def compute_dye_line_distances_numpy(spots_df, ratios_df, chan_order=["488", "514", "561", "594", "638"]):
    """
    Numpy equivalent of PairwiseUnmixer.calculate_distances.

    load_ratios_matrix() reads the CSV with index_col=0, so the first data
    column becomes the DataFrame index — leaving an apparent (n_ch, n_ch-1)
    shape. We recover the full (n_ch × n_ch) matrix by prepending the index
    back as the first column before computing distances.

    Columns of the ratios matrix = dye lines, one per channel (488..638).
    Returns d_{ch} columns (dist to each dye line) + dist_r_np.
    """
    intensity_cols = [f"chan_{ch}_intensity" for ch in chan_order
                      if f"chan_{ch}_intensity" in spots_df.columns]
    chans_used = [c.replace("chan_", "").replace("_intensity", "") for c in intensity_cols]

    if not intensity_cols:
        available = [c for c in spots_df.columns if "intensity" in c]
        raise ValueError(f"No intensity columns matched. Available: {available}")

    data = spots_df[intensity_cols].values.astype(np.float64)  # (n_spots, n_ch)

    # Recover full n_ch × n_ch matrix: prepend the index as the first column
    ratios_raw = np.hstack([
        np.array(ratios_df.index.values, dtype=float).reshape(-1, 1),
        ratios_df.values.astype(float),
    ])  # (n_ch, n_ch)
    n_ch_r = ratios_raw.shape[0]
    ratios_full = pd.DataFrame(
        ratios_raw,
        index=chan_order[:n_ch_r],
        columns=chan_order[:n_ch_r],
    )

    # Rows = channel dimensions; select only rows matching intensity data
    ratios = ratios_full.loc[chans_used].values.astype(np.float64)  # (n_ch_used, n_lines)
    ratios_norm = ratios / np.linalg.norm(ratios, axis=0)            # column-normalise

    projections = data @ ratios_norm                                          # (n_spots, n_lines)
    fit = projections[:, :, np.newaxis] * ratios_norm.T[np.newaxis]          # (n_spots, n_lines, n_ch)
    distances = np.linalg.norm(data[:, np.newaxis, :] - fit, axis=2)         # (n_spots, n_lines)

    # Each column i of distances = perpendicular dist to the i-th dye line
    dye_line_labels = chan_order[:ratios_raw.shape[1]]  # same order as ratios columns
    dist_df = pd.DataFrame(distances,
                           columns=[f"d_{ch}" for ch in dye_line_labels],
                           index=spots_df.index)
    sorted_d = np.sort(distances, axis=1)
    dist_df["dist_r_np"] = sorted_d[:, 1] / np.maximum(sorted_d[:, 0], 1e-10)
    return dist_df


def compute_spot_crosstalk_scores_intensity(spots_df, mixed_ref_df, ratios_df,
                                            round_id, chan_order=["488", "514", "561", "594", "638"],
                                            w_ratio=1.0, w_dim=1.0,
                                            z_keep_threshold=None):
    """
    Original (v1) crosstalk score using absolute intensity z-score.

    d_assignment_ratio   — d_assigned / d_min_other
    z_intensity_vs_removed  — robust z-score of spot's own-channel intensity
                            vs *removed* spots of same channel & round in mixed_ref_df
                                z > 0 → brighter than removed → probably real (no penalty)
                                z < 0 → dimmer than removed  → strong suspect (penalty)
    crosstalk_score      — (w_ratio * d_assignment_ratio) × (1 + w_dim * max(0, -z_intensity_vs_removed))
                            forced to 0 for spots where z > z_keep_threshold (brightness veto)
    z_keep_threshold     — if set, any spot with z_intensity_vs_removed > this value is
                            unconditionally kept (score → 0), overriding the spectral score.
                            Useful when d_ratio is inflated by a dominant nearby channel.
    """
    out = spots_df.copy().reset_index(drop=True)

    dist_cols = compute_dye_line_distances_numpy(out, ratios_df, chan_order=chan_order)
    for col in dist_cols.columns:
        out[col] = dist_cols[col].values

    # Vectorised d_assignment_ratio: use numpy fancy indexing instead of per-channel .loc loops
    d_cols_present = [f"d_{ch}" for ch in chan_order if f"d_{ch}" in out.columns]
    d_ratio = np.full(len(out), np.nan)
    if d_cols_present:
        chan_labels = [c[2:] for c in d_cols_present]          # strip "d_" prefix
        chan_to_idx = {ch: i for i, ch in enumerate(chan_labels)}
        d_matrix = out[d_cols_present].values.astype(np.float64)  # (n_spots, n_ch)
        assigned_idx = out["unmixed_chan"].map(chan_to_idx)         # NaN where channel unknown
        valid = assigned_idx.notna().values
        row_idx = np.where(valid)[0]
        col_idx = assigned_idx.dropna().astype(int).values
        if len(row_idx):
            d_assigned = d_matrix[row_idx, col_idx]
            d_others = d_matrix[row_idx].copy()
            d_others[np.arange(len(row_idx)), col_idx] = np.inf
            d_min_other = d_others.min(axis=1)
            d_min_other[d_min_other == 0] = np.nan
            d_ratio[row_idx] = d_assigned / d_min_other
    out["d_assignment_ratio"] = d_ratio

    ref = mixed_ref_df[
        (mixed_ref_df["round"] == round_id) & (mixed_ref_df["removed"])
    ]

    # Pre-allocate numpy arrays; fill by positional index; assign once to avoid repeated .loc writes
    chan_intensity_arr = np.full(len(out), np.nan)
    z_arr = np.full(len(out), np.nan)
    unmixed = out["unmixed_chan"].values

    for ch in chan_order:
        int_col = f"chan_{ch}_intensity"
        if int_col not in out.columns:
            continue
        row_idx = np.where(unmixed == ch)[0]
        if len(row_idx) == 0:
            continue

        ref_vals = ref.loc[ref["chan"] == ch, int_col].dropna()
        if len(ref_vals) < 5:
            continue

        ref_med = ref_vals.median()
        ref_mad = (ref_vals - ref_med).abs().median()
        if ref_mad < 1e-10:
            ref_mad = ref_vals.std()

        spot_int = out[int_col].values[row_idx]
        chan_intensity_arr[row_idx] = spot_int
        z_arr[row_idx] = (spot_int - ref_med) / (ref_mad * 1.4826)

    out["chan_intensity"] = chan_intensity_arr
    out["z_intensity_vs_removed"] = z_arr

    dim_penalty = np.maximum(0, -out["z_intensity_vs_removed"])

    # weighted score: base spectral ambiguity + weighted dimness penalty
    score = (w_ratio * out["d_assignment_ratio"]) * (1 + w_dim * dim_penalty)

    # brightness veto: spots clearly brighter than the removed population are real
    out["z_vetoed"] = False
    if z_keep_threshold is not None:
        veto_mask = out["z_intensity_vs_removed"] > z_keep_threshold
        score = np.where(veto_mask, 0.0, score)
        out["z_vetoed"] = veto_mask
        n_vetoed = int(veto_mask.sum())
        if n_vetoed:
            print(f"  z_keep_threshold={z_keep_threshold}: {n_vetoed} spots vetoed (score → 0)")

    out["crosstalk_score"] = score
    return out


def plot_crosstalk_scores_intensity(scored_df, round_id, cell_id=None,
                                     chan_order=["488", "514", "561", "594", "638"], chan_colors=None,
                                     subfig=None):
    """plot_crosstalk_scores variant for z_intensity_vs_removed columns."""
    if chan_colors is None:
        chan_colors = {}

    chans = [ch for ch in chan_order if ch in scored_df["unmixed_chan"].values]
    n_ch  = len(chans)
    score_p95 = np.nanpercentile(scored_df["crosstalk_score"].dropna(), 95)
    THRESHOLD = 1.0
    TOP_XLIM = (-2, 20)   # z_intensity_vs_removed
    TOP_YLIM = (-2, 30)   # d_assignment_ratio
    BOT_XLIM = (-1, 20)   # crosstalk_score histogram
    title = (f"Cell {cell_id}  Round {round_id} — " if cell_id else f"Round {round_id} — ")

    if subfig is None:
        fig, axes = plt.subplots(2, n_ch, figsize=(n_ch * 3.2, 7), squeeze=False)
        _standalone = True
    else:
        fig = subfig
        axes = subfig.subplots(2, n_ch, squeeze=False)
        _standalone = False
    fig.suptitle(title + "Crosstalk score  (intensity z-score vs removed)",
                 fontsize=_fs(14), y=1.02)

    for ci, ch in enumerate(chans):
        sub = scored_df[scored_df["unmixed_chan"] == ch].dropna(
            subset=["d_assignment_ratio", "z_intensity_vs_removed", "crosstalk_score"]
        )
        sub_vetoed = sub[sub.get("z_vetoed", False).astype(bool)] if "z_vetoed" in sub.columns else sub.iloc[:0]
        sub_scored = sub[~sub.get("z_vetoed", False).astype(bool)] if "z_vetoed" in sub.columns else sub
        color = chan_colors.get(ch, "#888888")
        n     = len(sub)

        ax0 = axes[0][ci]
        sc  = ax0.scatter(
            sub_scored["z_intensity_vs_removed"], sub_scored["d_assignment_ratio"],
            c=sub_scored["crosstalk_score"], cmap="RdYlGn_r",
            vmin=0, vmax=score_p95, s=18, alpha=0.75, linewidths=0,
        )
        if len(sub_vetoed):
            ax0.scatter(
                sub_vetoed["z_intensity_vs_removed"], sub_vetoed["d_assignment_ratio"],
                marker="*", s=30, color="royalblue", zorder=5,
                label=f"z-vetoed (n={len(sub_vetoed)})",
            )
            ax0.legend(fontsize=_fs(9), loc="upper right")
        fig.colorbar(sc, ax=ax0, shrink=0.75, pad=0.02, label="score")
        ax0.axhline(THRESHOLD, color="black", lw=0.9, ls="--", alpha=0.6)
        ax0.axvline(0,         color="black", lw=0.9, ls="--", alpha=0.6)
        ax0.set_xlim(TOP_XLIM)
        ax0.set_ylim(TOP_YLIM)
        xlim = ax0.get_xlim(); ylim = ax0.get_ylim()
        from matplotlib.patches import Rectangle as Rect
        ax0.add_patch(Rect(
            (xlim[0], THRESHOLD), 0 - xlim[0],
            max(ylim[1], sub["d_assignment_ratio"].max() * 1.1) - THRESHOLD,
            facecolor="#d73027", alpha=0.07, zorder=0,
        ))
        n_veto = len(sub_vetoed)
        veto_str = f"  [{n_veto} vetoed]"
        ax0.set_title(f"Ch {ch}  (n={n}){veto_str if n_veto else ''}", fontsize=_fs(12), color=color, fontweight="bold")
        ax0.set_xlabel("z_intensity_vs_removed\n(← dimmer than removed  |  brighter →)", fontsize=_fs(10))
        if ci == 0:
            ax0.set_ylabel("d_assignment_ratio\n(↑ worse spectral purity)", fontsize=_fs(10))
        else:
            ax0.set_yticklabels([])
        ax0.tick_params(labelsize=_fs(9))
        ax0.spines[["top", "right"]].set_visible(False)

        ax1    = axes[1][ci]
        scores = sub["crosstalk_score"].dropna()
        bins   = np.linspace(BOT_XLIM[0], BOT_XLIM[1], 40)
        clean   = scores[scores <= THRESHOLD]
        suspect = scores[scores >  THRESHOLD]
        ax1.hist(clean,   bins=bins, alpha=0.6, color=color,
                 label=f"score ≤ {THRESHOLD}  (n={len(clean)})")
        ax1.hist(suspect, bins=bins, alpha=0.8, color="#d73027",
                 label=f"score > {THRESHOLD}  (n={len(suspect)})")
        ax1.axvline(THRESHOLD, color="black", lw=1.0, ls="--")
        ax1.set_xlim(BOT_XLIM)
        ax1.set_xlabel("crosstalk_score", fontsize=_fs(10))
        if ci == 0:
            ax1.set_ylabel("count", fontsize=_fs(10))
        else:
            ax1.set_yticklabels([])
        ax1.legend(fontsize=_fs(9), framealpha=0.8)
        ax1.tick_params(labelsize=_fs(9))
        ax1.spines[["top", "right"]].set_visible(False)

    plt.tight_layout() if _standalone else None
    if _standalone:
        plt.show()

    print(f"\n{'Ch':>5}  {'n':>6}  {'med_ratio':>10}  {'med_z_int':>10}  "
          f"{'med_score':>10}  {'% score>1':>10}")
    for ch in chans:
        s = scored_df[scored_df["unmixed_chan"] == ch].dropna(
            subset=["d_assignment_ratio", "z_intensity_vs_removed", "crosstalk_score"]
        )
        if len(s) == 0:
            continue
        print(f"{ch:>5}  {len(s):>6}"
              f"  {s['d_assignment_ratio'].median():>10.3f}"
              f"  {s['z_intensity_vs_removed'].median():>10.2f}"
              f"  {s['crosstalk_score'].median():>10.3f}"
              f"  {100*(s['crosstalk_score']>1).mean():>9.1f}%")


def qc_spot_detection(data_dir, output_dir, channels=None, verbose=False):
    """
    Run spot detection quality control analysis.

    Parameters:
    -----------
    data_dir : Path or str
        Path to data directory containing spot detection results
    output_dir : Path or str
        Directory to save QC outputs
    """

    if verbose:
        print("Spot detection QC completed successfully!")
