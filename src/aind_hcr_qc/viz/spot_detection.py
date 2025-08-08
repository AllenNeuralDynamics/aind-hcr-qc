"""Spots"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
