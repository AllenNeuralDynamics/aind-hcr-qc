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
