"""Integrated datasets"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aind_hcr_qc.utils.utils import saveable_plot
import aind_hcr_qc.constants as constants

# channel name → intensity column
_CHAN_INTENSITY_COL = {
    "488": "chan_488_intensity",
    "514": "chan_514_intensity",
    "561": "chan_561_intensity",
    "594": "chan_594_intensity",
    "638": "chan_638_intensity",
}


def add_unmixed_channel_intensity(
    spots_df: pd.DataFrame,
    chan_col: str = "unmixed_chan",
    intensity_threshold: float | None = 20.0,
) -> pd.DataFrame:
    """
    Add an '_intensity' column containing each spot's intensity in its own
    channel, then optionally filter to spots above a threshold.

    Parameters
    ----------
    spots_df : pd.DataFrame
        Spots dataframe with a channel column and chan_*_intensity columns.
    chan_col : str
        Column whose values identify which channel intensity to pull for each
        spot. Defaults to "unmixed_chan"; could also be e.g. "chan".
    intensity_threshold : float or None
        Drop rows where _intensity <= threshold. Pass None to skip filtering.

    Returns
    -------
    pd.DataFrame
        Input dataframe with '_intensity' column added and filtered rows removed.
    """
    intensity = pd.Series(np.nan, index=spots_df.index)
    for chan, col in _CHAN_INTENSITY_COL.items():
        if col in spots_df.columns:
            mask = spots_df[chan_col].astype(str) == chan
            intensity[mask] = spots_df.loc[mask, col]

    out = spots_df.copy()
    out["_intensity"] = intensity
    out = out.dropna(subset=["_intensity"])

    if intensity_threshold is not None:
        out = out[out["_intensity"] > intensity_threshold]

    return out


@saveable_plot()
def plot_intensity_violins(
    spots_df: pd.DataFrame,
    order: str = "round_chan",
    chan_col: str = "unmixed_chan",
    intensity_threshold: float | None = 20.0,
    n_sample: int = 25_000,
    ax=None,
    figsize=None,
):
    """
    Violin plots of per-spot intensity for each unmixed gene.

    Parameters
    ----------
    spots_df : pd.DataFrame
        Spots dataframe with columns including unmixed_chan, unmixed_gene,
        rd_ch_unmixed_gene, and chan_*_intensity columns.
    order : str
        "round_chan" — x-axis is rd_ch_unmixed_gene sorted by round then
                       channel; violins colored by channel.
        "alpha"      — x-axis is unmixed_gene sorted alphabetically.
    chan_col : str
        Column used to look up which channel intensity to plot per spot.
        Defaults to "unmixed_chan"; could also be e.g. "chan".
    intensity_threshold : float or None
        Passed to add_unmixed_channel_intensity; drops spots at or below this
        value. Default 20 removes the sub-threshold floor pile-up.
    n_sample : int
        Max spots per group passed to the KDE. Default 25000 is plenty for
        an accurate violin shape and keeps rendering fast.
    ax : matplotlib.axes.Axes, optional
    figsize : tuple, optional
    """
    x_col = "rd_ch_unmixed_gene" if order == "round_chan" else "unmixed_gene"
    group_col = x_col

    # Only keep columns we actually need — avoids copying 29M rows in full
    intensity_cols = [c for c in _CHAN_INTENSITY_COL.values() if c in spots_df.columns]
    keep = [chan_col, "round", x_col] + intensity_cols
    df = add_unmixed_channel_intensity(spots_df[keep], chan_col=chan_col, intensity_threshold=intensity_threshold)

    # Subsample per group — KDE only needs thousands of points, not millions.
    # Avoid groupby.apply: pandas 3.0 excludes the groupby key column from the
    # group DataFrames passed to apply, which loses group_col from the result.
    rng = np.random.default_rng(0)
    sampled_idx = np.concatenate([
        rng.choice(idx, min(len(idx), n_sample), replace=False)
        for idx in df.groupby(group_col, sort=False).groups.values()
    ])
    df = df.loc[sampled_idx].reset_index(drop=True)
    print(df)

    if order == "round_chan":
        x_col = "rd_ch_unmixed_gene"
        cats = (
            df[["round", chan_col, x_col]]
            .drop_duplicates()
            .sort_values(["round", chan_col])
            [x_col]
            .tolist()
        )
        # color each violin by its channel (middle token of "Rx-NNN-Gene")
        palette = {
            label: constants.Z1_CHANNEL_CMAP_SOFT.get(str(label).split("-")[1], "#999999")
            for label in cats
        }
        xlabel = "Round – Channel – Gene"
    else:
        x_col = "unmixed_gene"
        cats = sorted(df[x_col].dropna().unique())
        palette = None
        xlabel = "Gene"

    if figsize is None:
        figsize = (max(10, len(cats) * 0.8), 5)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    sns.violinplot(
        data=df,
        x=x_col,
        y="_intensity",
        order=cats,
        palette=palette,
        inner="quartile",
        density_norm="count",
        cut=0,

        ax=ax,
    )

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Intensity (log scale)")
    ax.set_title("Spot intensities by " + ("round–channel–gene" if order == "round_chan" else "gene"))
    ax.tick_params(axis="x", rotation=90)

    return ax


def plot_gene_spot_count_pairplot(
    cxg_wide: pd.DataFrame,
    genes_to_plot: list[str] | None = None,
    title_prefix: str = "",
    n_sample: int = 10_000,
    clip_pct: float = 99.9999,
) -> plt.Figure:
    """
    Pairwise spot-count scatter plot across genes, one point per cell.

    Parameters
    ----------
    cxg_wide : pd.DataFrame
        Wide-format aggregated CXG from ``pw_dataset.load_aggregated_cxg()``.
        Columns are ``{round}-{channel}-{gene}`` labels; index is ``cell_id``.
    genes_to_plot : list[str] or None
        Subset of gene names to include. ``None`` uses all genes present.
    title_prefix : str
        Prepended to the figure suptitle (e.g. ``"755252 - "``).
    n_sample : int
        Max number of cells to plot. Cells are sampled randomly when exceeded.
    clip_pct : float
        Percentile at which to clip outlier counts per gene before plotting.

    Returns
    -------
    plt.Figure
    """
    # Aggregate round×channel columns down to per-gene totals.
    # Column names are "Rn-chan-Gene"; gene is everything after the last "-".
    gene_map: dict[str, list[str]] = {}
    for col in cxg_wide.columns:
        gene = col.split("-")[-1]
        gene_map.setdefault(gene, []).append(col)

    gene_df = pd.DataFrame(
        {gene: cxg_wide[cols].sum(axis=1) for gene, cols in gene_map.items()},
        index=cxg_wide.index,
    )

    if genes_to_plot is not None:
        present = [g for g in genes_to_plot if g in gene_df.columns]
        missing = [g for g in genes_to_plot if g not in gene_df.columns]
        if missing:
            print(f"  [pairplot] genes not found in CXG, skipping: {missing}")
        gene_df = gene_df[present]

    # Clip outliers then subsample
    gene_df = gene_df.clip(upper=gene_df.quantile(clip_pct / 100), axis=1)
    if len(gene_df) > n_sample:
        gene_df = gene_df.sample(n_sample, random_state=42)

    n_genes = gene_df.shape[1]
    label = f"{n_genes} genes" if genes_to_plot is None else f"{n_genes} selected genes"

    g = sns.pairplot(
        gene_df,
        corner=True,
        plot_kws={"s": 10, "alpha": 0.4, "edgecolor": "none", "rasterized": True},
        diag_kind="kde",
        diag_kws={"fill": True},
    )
    title = f"{title_prefix}Pairwise Gene Spot Counts per Cell — {label}"
    g.figure.suptitle(title.strip(), y=1.02, fontsize=14)
    plt.tight_layout()

    return g.figure
