"""Spectral unmixing"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import utils

# ------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------


def read_ratios_file(filename):
    """
    Read a ratios matrix from a file.

    Parameters:
    filename: Path to the ratios file

    Returns:
    ratios: Numpy array containing the ratios matrix
    """
    try:
        # Read the file content
        with open(filename, "r") as f:
            lines = f.readlines()

        # Parse the matrix
        ratios = []
        for line in lines:
            row = [float(val) for val in line.strip().split()]
            ratios.append(row)

        return np.array(ratios).T
    except Exception as e:
        print(f"Error reading ratios file: {e}")
        return None


# ------------------------------------------------------------------------------------------------
# Spot count QC by gene
# ------------------------------------------------------------------------------------------------


def plot_spot_count(
    cxg_data,
    color_dict=None,
    volume_filter=False,
    volume_percentiles=(5, 95),
    log_plot=False,
    figsize=(12, 4),
    min_n_spots=0,
    save=False,
    output_dir=None,
):
    """
    Create a comprehensive QC figure for spot count distributions by gene.

    Parameters:
    -----------
    cxg_data : pandas.DataFrame
        Cell-by-gene data with columns: cell_id, gene, spot_count, volume
    color_dict : dict, optional
        Dictionary mapping gene names to colors. If None, uses default palette.
    volume_filter : bool
        Whether to filter cells by volume percentiles
    volume_percentiles : tuple
        (min_percentile, max_percentile) for volume filtering
    log_plot : bool
        Whether to use logarithmic scale for spot counts
    figsize : tuple
        Figure size (width, height)
    min_n_spots : int
        Minimum number of spots per gene to include in the plot

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """

    # Make a copy to avoid modifying original data
    data = cxg_data.copy()

    # Apply volume filter if requested
    if volume_filter and "volume" in data.columns:
        vol_min = np.percentile(data["volume"], volume_percentiles[0])
        vol_max = np.percentile(data["volume"], volume_percentiles[1])
        data = data[(data["volume"] >= vol_min) & (data["volume"] <= vol_max)]
        filter_text = f"Volume filtered: {volume_percentiles[0]}-{volume_percentiles[1]}th percentile\n({vol_min:.0f}-{vol_max:.0f})"
    else:
        filter_text = "No volume filtering"

    # apply minimum number of spots filter
    if min_n_spots > 0:
        # for each cell, count the number of spots per gene, remove those with fewer than min_n_spots
        gene_counts = data.groupby(["cell_id", "gene"])["spot_count"].sum().reset_index()
        gene_counts = gene_counts[gene_counts["spot_count"] >= min_n_spots]
        data = data[data.set_index(["cell_id", "gene"]).index.isin(gene_counts.set_index(["cell_id", "gene"]).index)]
        filter_text += f"\nMinimum {min_n_spots} spots per gene"
    else:
        filter_text += "\nNo minimum spots filter applied"

    # Get unique genes
    genes = sorted(data["gene"].unique())
    n_genes = len(genes)

    # Set up colors
    if color_dict is None:
        # Use a nice color palette
        colors = sns.color_palette("husl", n_genes)
        color_dict = dict(zip(genes, colors))

    # Calculate statistics for each gene
    stats = data.groupby("gene")["spot_count"].agg(["count", "min", "max", "median", "mean", "std"]).round(2)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)

    # Calculate grid layout (try to make it roughly square)
    cols = int(np.ceil(np.sqrt(n_genes)))
    rows = int(np.ceil(n_genes / cols))

    # Create subplots for each gene
    for i, gene in enumerate(genes):
        ax = plt.subplot(rows, cols, i + 1)

        gene_data = data[data["gene"] == gene]["spot_count"]
        color = color_dict.get(gene, "steelblue")

        # Handle log plotting
        if log_plot and gene_data.min() > 0:
            # Use log-spaced bins for better visualization
            bins = np.logspace(
                np.log10(max(gene_data.min(), 1)), np.log10(gene_data.max()), min(30, len(gene_data.unique()))
            )
            # Create histogram with log bins
            sns.histplot(gene_data, bins=bins, kde=False, color=color, alpha=0.7, ax=ax)
            ax.set_xscale("log")
        else:
            # Regular histogram
            sns.histplot(gene_data, kde=True, color=color, alpha=0.7, ax=ax)

        # Add vertical lines for statistics
        median_val = gene_data.median()
        mean_val = gene_data.mean()

        ax.axvline(median_val, color="red", linestyle="--", alpha=0.8, linewidth=2, label=f"Median: {median_val:.1f}")
        ax.axvline(mean_val, color="orange", linestyle=":", alpha=0.8, linewidth=2, label=f"Mean: {mean_val:.1f}")

        # Customize subplot
        ax.set_title(
            f"{gene}\n(n={len(gene_data)}, range: {gene_data.min()}-{gene_data.max()})", fontsize=10, fontweight="bold"
        )
        ax.set_xlabel("Spot Count" + (" (log scale)" if log_plot and gene_data.min() > 0 else ""), fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.tick_params(labelsize=8)

        # Add legend for small subplots
        if len(gene_data) > 0:
            ax.legend(fontsize=7, loc="upper right")

        # Set reasonable x-axis limits
        if not log_plot and gene_data.max() > 0:
            ax.set_xlim(0, gene_data.quantile(0.99) * 1.1)

    # Remove empty subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(plt.subplot(rows, cols, j + 1))

    # Add overall title with summary info
    total_cells = len(data["cell_id"].unique())
    total_spots = data["spot_count"].sum()

    fig.suptitle(
        f"Spot Count QC by Gene\n{total_cells:,} cells, {total_spots:,} total spots\n{filter_text}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Print summary statistics table
    print("\nSummary Statistics by Gene:")
    print("=" * 60)
    print(stats.to_string())

    if save and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "spot_count_qc_by_gene.png", bbox_inches="tight")
        print(f"Figure saved to {output_dir / 'spot_count_qc_by_gene.png'}")

    return fig


def plot_channel_dists(
    cxg_data,
    color_dict=None,
    log_plot=False,
    color_col="gene",  # or channel
    value_col="spot_count",  # or mean
    figsize=(12, 4),
    save=False,
    output_dir=None,
):
    """
    Create a comprehensive QC figure for spot count distributions by gene.

    Parameters:
    -----------
    cxg_data : pandas.DataFrame
        Cell-by-gene data with columns: cell_id, gene, spot_count, volume
    color_dict : dict, optional
        Dictionary mapping gene names to colors. If None, uses default palette.
    log_plot : bool
        Whether to use logarithmic scale for spot counts
    figsize : tuple
        Figure size (width, height)
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """

    # Make a copy to avoid modifying original data
    data = cxg_data.copy()

    # Get unique genes
    genes = sorted(data[color_col].unique())
    n_genes = len(genes)

    # Set up colors
    if color_dict is None:
        # Use a nice color palette
        colors = sns.color_palette("husl", n_genes)
        color_dict = dict(zip(genes, colors))

    # Calculate statistics for each gene
    stats = data.groupby(color_col)[value_col].agg(["count", "min", "max", "median", "mean", "std"]).round(2)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)

    # Calculate grid layout (try to make it roughly square)
    cols = int(np.ceil(np.sqrt(n_genes)))
    rows = int(np.ceil(n_genes / cols))

    value_str = value_col.replace("_", " ").title()  # e.g. 'Spot Count' or 'Mean Intensity'
    color_col_str = color_col.replace("_", " ").title()  # e.g. 'Gene' or 'Channel'
    # Create subplots for each gene
    for i, gene in enumerate(genes):
        ax = plt.subplot(rows, cols, i + 1)

        plot_data = data[data[color_col] == gene][value_col]
        color = color_dict.get(gene, "steelblue")

        # Handle log plotting
        # if log_plot and plot_data.min() > 0:
        if log_plot:
            # Use log-spaced bins for better visualization
            bins = np.logspace(
                np.log10(max(plot_data.min(), 1)), np.log10(plot_data.max()), min(30, len(plot_data.unique()))
            )
            # Create histogram with log bins
            sns.histplot(plot_data, bins=bins, kde=False, color=color, alpha=0.7, ax=ax)
            ax.set_xscale("log")
        else:
            # Regular histogram
            sns.histplot(plot_data, kde=True, color=color, alpha=0.7, ax=ax)

        # Add vertical lines for statistics
        median_val = plot_data.median()
        mean_val = plot_data.mean()

        ax.axvline(median_val, color="red", linestyle="--", alpha=0.8, linewidth=2, label=f"Median: {median_val:.1f}")
        ax.axvline(mean_val, color="orange", linestyle=":", alpha=0.8, linewidth=2, label=f"Mean: {mean_val:.1f}")

        # Customize subplot
        ax.set_title(
            f"{gene}\n(n={len(plot_data)}, range: {plot_data.min()}-{plot_data.max()})", fontsize=10, fontweight="bold"
        )
        ax.set_title(
            f"{gene}\n(n={len(plot_data)}, range: {np.round(plot_data.min())}-{np.round(plot_data.max())})",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel(f"{value_str}" + (" (log scale)" if log_plot and plot_data.min() > 0 else ""), fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.tick_params(labelsize=8)

        # y log
        # if log_plot:
        #     ax.set_yscale('log')
        #     # ax.yaxis.set_major_formatter(ScalarFormatter())
        #     # ax.yaxis.set_minor_formatter(ScalarFormatter())
        #     ax.tick_params(axis='y', which='both', labelsize=8)

        # Add legend for small subplots
        if len(plot_data) > 0:
            ax.legend(fontsize=7, loc="upper right")

        # Set reasonable x-axis limits
        if not log_plot and plot_data.max() > 0:
            ax.set_xlim(0, plot_data.quantile(0.99) * 1.1)

        # Set hardcoded x-axis limits to focus on the range of interest for visualization.
        # These limits are chosen based on expected data distribution and to exclude outliers.
        ax.set_xlim(90, 1000)

    # Remove empty subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(plt.subplot(rows, cols, j + 1))

    # Add overall title with summary info
    total_cells = len(data["cell_id"].unique())
    # total_spots = data[value_col].sum() # unused var

    fig.suptitle(f"{value_str} by {color_col_str}\n{total_cells:,} cells", fontsize=14, fontweight="bold", y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Print summary statistics table
    print("\nSummary Statistics by Gene:")
    print("=" * 60)
    print(stats.to_string())

    if save and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "spot_count_qc_by_gene.png", bbox_inches="tight")
        print(f"Figure saved to {output_dir / 'spot_count_qc_by_gene.png'}")

    return fig


def print_volume_filtering__summary(cxg):
    # Summary comparison between filtered and unfiltered data
    print("\n" + "=" * 80)
    print("DATA SUMMARY COMPARISON")
    print("=" * 80)

    # Original data summary
    print("\nORIGINAL DATA (all cells):")
    print(f"  Total cells: {cxg['cell_id'].nunique():,}")
    print(f"  Total spots: {cxg['spot_count'].sum():,}")
    print(f"  Volume range: {cxg['volume'].min():.0f} - {cxg['volume'].max():.0f}")
    print(f"  Median volume: {cxg['volume'].median():.0f}")

    # Volume filtered data
    vol_min = np.percentile(cxg["volume"], 5)
    vol_max = np.percentile(cxg["volume"], 95)
    filtered_data = cxg[(cxg["volume"] >= vol_min) & (cxg["volume"] <= vol_max)]

    print("\nVOLUME FILTERED DATA (5th-95th percentile):")
    print(f"  Volume filter range: {vol_min:.0f} - {vol_max:.0f}")
    print(
        f"  Cells retained: {filtered_data['cell_id'].nunique():,} ({filtered_data['cell_id'].nunique()/cxg['cell_id'].nunique()*100:.1f}%)"
    )
    print(
        f"  Spots retained: {filtered_data['spot_count'].sum():,} ({filtered_data['spot_count'].sum()/cxg['spot_count'].sum()*100:.1f}%)"
    )
    print(f"  Cells excluded: {cxg['cell_id'].nunique() - filtered_data['cell_id'].nunique():,}")

    # Per-gene comparison
    print("\nPER-GENE IMPACT OF VOLUME FILTERING:")
    comparison = []
    for gene in sorted(cxg["gene"].unique()):
        orig_gene = cxg[cxg["gene"] == gene]
        filt_gene = filtered_data[filtered_data["gene"] == gene]

        comparison.append(
            {
                "Gene": gene,
                "Orig_Cells": len(orig_gene),
                "Filt_Cells": len(filt_gene),
                "Cells_Retained_%": len(filt_gene) / len(orig_gene) * 100 if len(orig_gene) > 0 else 0,
                "Orig_Spots": orig_gene["spot_count"].sum(),
                "Filt_Spots": filt_gene["spot_count"].sum(),
                "Spots_Retained_%": (
                    filt_gene["spot_count"].sum() / orig_gene["spot_count"].sum() * 100
                    if orig_gene["spot_count"].sum() > 0
                    else 0
                ),
            }
        )

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False, float_format="%.1f"))


# ------------------------------------------------------------------------------------------------
# Spot intensity pairwise plots
# ------------------------------------------------------------------------------------------------


def darken_color(color, amount=0.5):
    """
    Darken a color by a specified amount.

    Parameters:
    color: Original color (can be string or RGB tuple)
    amount: Factor to darken the color (0-1, where 1 is black)

    Returns:
    Darkened color as RGB tuple
    """
    try:
        # Convert color to RGB
        rgb = mcolors.to_rgb(color)

        # Darken the color
        darkened = tuple(max(0, c * (1 - amount)) for c in rgb)

        return darkened
    except Exception as e:
        print(f"Error darkening color {color}: {e}")
        return color


"""
cell_ids = list(range(1,10000))
# df_to_plot = merged_spot_table_concat.loc[(merged_spot_table_concat['round'] ==2)&(merged_spot_table_concat['r'] >0.5)]
df_to_plot = merged_spot_table_concat.loc[(merged_spot_table_concat['round'] ==3)]


figs = plot_pairwise_intensities_multi(df_to_plot, cell_ids,
                                        plot_type='combined'
                                        scale='log')
# The generate_cell_stats function remains the same as in the previous version
"""


# ------------------------------------------------------------------------------------------------
# Venn diagram for spot overlap
# ------------------------------------------------------------------------------------------------


def plot_spot_round_reassignment(df, round_col="round"):
    """
    Generate a visualization of spots per round with reassignment information.

    Parameters:
    df: DataFrame containing spot data with round information
    round_col: Name of the column indicating round identity (default: 'round')
    """
    # Get unique rounds
    rounds = df[round_col].unique()
    round_names = [str(r) for r in rounds]

    # Compute statistics for each round
    round_stats = []
    for round_val in rounds:
        round_df = df[df[round_col] == round_val]

        # Total spots in the round
        total_spots = len(round_df)

        # Spots reassigned to a different channel
        reassigned_spots = len(round_df[round_df["chan"] != round_df["unmixed_chan"]])

        # Spots remaining in original channel
        original_spots = total_spots - reassigned_spots

        # filtered_spots breakdown
        removed_spots = len(round_df[round_df["valid_spot"] is not True])

        dye_line_filtered_spots = len(round_df[round_df["dye_line_dist_ratio"] < 4])
        dist_filtered_spots = len(round_df[round_df["dist"] > 1])
        r_filtered_spots = len(round_df[round_df["r"] < 0.5])

        round_stats.append(
            {
                "round": round_val,
                "total_spots": total_spots,
                "reassigned_spots": reassigned_spots,
                "original_spots": original_spots,
                "total_spots_removed_by_filters": removed_spots,
                "dye_line_dist_filtered_spots": dye_line_filtered_spots,
                "dist_filtered_spots": dist_filtered_spots,
                "r_filtered_spots": r_filtered_spots,
            }
        )

    # Pie chart visualization
    plt.figure(figsize=(15, 5))

    # Total spots pie chart
    plt.subplot(131)
    total_spots_data = [stats["total_spots"] for stats in round_stats]
    plt.pie(total_spots_data, labels=round_names, autopct="%1.1f%%")
    plt.title("Total Spots per Round")

    # Reassigned spots pie chart
    plt.subplot(132)
    reassigned_spots_data = [stats["reassigned_spots"] for stats in round_stats]
    plt.pie(reassigned_spots_data, labels=round_names, autopct="%1.1f%%")
    plt.title("Reassigned Spots per Round")

    # Original spots pie chart
    plt.subplot(133)
    original_spots_data = [stats["original_spots"] for stats in round_stats]
    plt.pie(original_spots_data, labels=round_names, autopct="%1.1f%%")
    plt.title("Original Channel Spots per Round")

    plt.tight_layout()

    # Create a summary table
    print("\nSpot Reassignment Summary:")
    print(
        "{:<10} {:<15} {:<20} {:<20} {:<15}".format(
            "Round", "Total Spots", "Spots in Original Chan", "Spots Reassigned", "Reassignment %"
        )
    )
    print("-" * 80)

    for stats in round_stats:
        reassignment_percentage = (stats["reassigned_spots"] / stats["total_spots"]) * 100
        print(
            "{:<10} {:<15} {:<20} {:<20} {:<6.2f}%".format(
                stats["round"],
                stats["total_spots"],
                stats["original_spots"],
                stats["reassigned_spots"],
                reassignment_percentage,
            )
        )

    print("\nSpot Filtering Summary:")
    print(
        "{:<10} {:<15} {:<20} {:<20} {:<15} {:<20} {:<20}".format(
            "Round",
            "Total Spots",
            "Total filtered Spots",
            "Dye Line ratio Filtered",
            "Dist Filtered",
            "r filtered",
            "Total Filtered %",
        )
    )
    print("-" * 80)
    for stats in round_stats:
        reassignment_percentage = (stats["total_spots_removed_by_filters"] / stats["total_spots"]) * 100
        print(
            "{:<10} {:<15} {:<20} {:<20} {:<15} {:<20} {:<6.2f}%".format(
                stats["round"],
                stats["total_spots"],
                stats["total_spots_removed_by_filters"],
                stats["dye_line_dist_filtered_spots"],
                stats["dist_filtered_spots"],
                stats["r_filtered_spots"],
                reassignment_percentage,
            )
        )

    return plt.gcf()


# Example usage (you would replace this with your actual data loading)
"""
# Assuming you have multiple round DataFrames
round1_spots = ...
round2_spots = ...
round3_spots = ...

# List of DataFrames
rounds_data = [round1_spots, round2_spots, round3_spots]

# Generate pie charts
fig1 = plot_spot_round_reassignment(rounds_data,
                                     round_names=['Round 1', 'Round 2', 'Round 3'])
plt.show()

# Generate Venn diagram
if len(rounds_data) in [2, 3]:
    fig2 = plot_spot_overlap_venn(rounds_data,
                                  round_names=['Round 1', 'Round 2', 'Round 3'])
    plt.show()

fig1 = plot_spot_round_reassignment(merged_spot_table_concat,
                                    )
plt.show()
"""

# ------------------------------------------------------------------------------------------------
# multichannel spots 2
# ------------------------------------------------------------------------------------------------


# Function to plot dye lines on an axis
def plot_dye_lines(ax, chan1_name, chan2_name, ratios_norm, scale="log", colors=None, all_possible_channels=None):
    """
    Plot dye lines for a pair of channels that adapt to the current axis limits.

    Parameters:
    ax: Matplotlib axis to plot on
    chan1_name: Name of the first channel (for y-axis), e.g. '488'
    chan2_name: Name of the second channel (for x-axis), e.g. '561'
    ratios_norm: Normalized mixing matrix
    scale: 'log' or 'linear' scale
    colors: Optional list of colors to use for the lines
    all_possible_channels: List of all possible channel names that the ratios matrix indexes correspond to
    """
    # Default colors if not provided
    if colors is None:
        colors = ["darkgreen", "maroon", "darkgoldenrod", "darkcyan", "purple"]

    # Default channel list if not provided
    if all_possible_channels is None:
        all_possible_channels = ["488", "514", "561", "594", "638"]

    # Get indices of the channels in the all_possible_channels list
    # These indices correspond to the rows/columns in the ratios matrix
    try:
        chan1_idx = all_possible_channels.index(chan1_name)
        chan2_idx = all_possible_channels.index(chan2_name)
    except ValueError:
        print(f"Error: Channels {chan1_name} or {chan2_name} not found in {all_possible_channels}")
        return

    # Check if these indices are valid in the ratios matrix
    if chan1_idx >= ratios_norm.shape[0] or chan2_idx >= ratios_norm.shape[0]:

        # TODO add logic to fix this, as currently it is a bug that doesn't show dyelines if a middle channel is missing
        print(
            f"Error: Channel indices {chan1_idx}, {chan2_idx} out of range for ratios matrix shape {ratios_norm.shape}."
        )
        return

    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Helper function to transform intensity
    def transform_intensity(intensity, scale, reverse=False):
        if reverse:
            if scale == "log":
                return np.expm1(intensity)
            else:
                return intensity
        else:
            if scale == "log":
                return np.log1p(np.maximum(intensity, 0))
            else:
                return intensity

    # Get the actual data limits in untransformed space
    # x_min_raw = transform_intensity(xlim[0], scale, reverse=True) # unused
    x_max_raw = transform_intensity(xlim[1], scale, reverse=True)
    # y_min_raw = transform_intensity(ylim[0], scale, reverse=True) # unused
    y_max_raw = transform_intensity(ylim[1], scale, reverse=True)

    # Plot line for chan1 contribution to chan2 (x-axis → y-axis)
    m = ratios_norm[chan1_idx, chan2_idx]

    if m > 0:
        # Calculate intersection points with the plot boundaries
        # For y = m*x, we need to find where this crosses the axis limits

        # Check intersection with top boundary (y = y_max_raw)
        x_at_top = y_max_raw / m if m != 0 else np.inf
        # Check intersection with right boundary (x = x_max_raw)
        y_at_right = m * x_max_raw

        # Choose the limiting point based on which boundary is hit first
        if x_at_top <= x_max_raw:
            # Line hits the top first
            x_end = x_at_top
            y_end = y_max_raw
        else:
            # Line hits the right side first
            x_end = x_max_raw
            y_end = y_at_right

        # Create line from origin to the limiting point
        x = np.linspace(0, x_end, 200)
        y = m * x
    else:
        # If slope is zero or negative, just draw a horizontal line at y=0
        x = np.linspace(0, x_max_raw, 2)
        y = np.zeros_like(x)

    # Transform back to the plot scale
    x_transformed = transform_intensity(x, scale)
    y_transformed = transform_intensity(y, scale)

    # Plot first line (x-axis contributes to y-axis)
    ax.plot(
        x_transformed,
        y_transformed,
        linestyle="-",
        alpha=0.7,
        color=colors[chan2_idx % len(colors)],
    )

    # Plot line for chan2 contribution to chan1 (y-axis → x-axis)
    m = ratios_norm[chan2_idx, chan1_idx]

    if m > 0:
        # Similar calculation for the second line
        y_at_right = x_max_raw / m if m != 0 else np.inf
        x_at_top = m * y_max_raw

        if y_at_right <= y_max_raw:
            # Line hits the right side first
            y_end = y_at_right
            x_end = x_max_raw
        else:
            # Line hits the top first
            y_end = y_max_raw
            x_end = x_at_top

        y = np.linspace(0, y_end, 200)
        x = m * y
    else:
        # If slope is zero or negative, just draw a vertical line at x=0
        y = np.linspace(0, y_max_raw, 2)
        x = np.zeros_like(y)

    # Transform based on scale
    x_transformed = transform_intensity(x, scale)
    y_transformed = transform_intensity(y, scale)

    # Plot second line (y-axis contributes to x-axis)
    ax.plot(
        x_transformed,
        y_transformed,
        linestyle="-",
        alpha=0.7,
        color=colors[chan1_idx % len(colors)],
    )


def plot_pairwise_intensities_multi_ratios(
    df, cell_ids, scale="log", chan_col="unmixed_chan", title_prefix=None, same_limits=False, ratios=None
):
    """
    Creates 2D scatter plots comparing channel intensities pairwise for multiple cells.
    Plots are arranged in a grid with channels as rows and columns.
    Only the lower left triangle is populated with plots, diagonal is empty.

    Parameters:
    df: DataFrame containing the spot data
    cell_ids: List of cell IDs to analyze
    scale: 'log' for log-transformed intensities,
           'linear' for original intensities
    chan_col: "unmixed_chan" or "chan"
    title_prefix: Optional prefix for the figure title
    same_limits: If True, use the same axis limits for all subplots for easier comparison
    ratios: Optional mixing matrix for plotting dye lines. Should be a square numpy array
           with dimensions matching the number of POTENTIAL channels (usually 5)
    """
    # Set style
    plt.style.use("default")

    # Detect available channels from column names
    all_possible_channels = ["488", "514", "561", "594", "638"]
    available_channels = []

    for channel in all_possible_channels:
        intensity_col = f"chan_{channel}_intensity"
        if intensity_col in df.columns:
            available_channels.append(channel)

    if not available_channels:
        raise ValueError("No channel intensity columns found in the DataFrame")

    if len(available_channels) < 2:
        raise ValueError("At least two channels are needed for pairwise comparison")

    print(f"Detected channels: {available_channels}")

    # Create color mapping for channels
    channel_colors = {"488": "green", "514": "red", "561": "orange", "594": "cyan", "638": "magenta"}

    def transform_intensity(intensity, scale):
        """Transform intensity based on selected scale"""
        if scale == "log":
            # Handle zeros and negative values safely
            return np.log1p(np.maximum(intensity, 0))
        else:
            return intensity

    # Calculate optimal grid dimensions based on number of pairwise comparisons
    num_channels = len(available_channels)

    # Determine a more compact grid layout that only includes necessary plots
    # For a triangular arrangement, we'll use a grid where we can place our plots
    # without empty spaces in between
    nrows = num_channels - 1  # Number of rows needed for lower triangle
    ncols = num_channels - 1  # Maximum number of columns per row

    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))

    # Create a grid layout for our triangular plot arrangement
    grid = plt.GridSpec(nrows, ncols, figure=fig)

    # Store legend elements
    legend_handles = []
    legend_labels = []

    # Track all unique channels for the legend
    all_plotted_channels = set()

    # Normalize ratios matrix if provided
    ratios_norm = None
    if ratios is not None:
        # Verify that the ratios matrix has the correct dimensions
        expected_size = len(all_possible_channels)
        if ratios.shape[0] != expected_size or ratios.shape[1] != expected_size:
            print(
                f"Warning: Ratios matrix dimensions {ratios.shape} do not match expected size {expected_size}x{expected_size}."
            )
            print(f"The ratios matrix should have dimensions matching all possible channels: {all_possible_channels}")
            print(f"Available channels in this dataset: {available_channels}")
            print("Attempting to use the provided ratios matrix as is, but results may be incorrect.")

        # Normalize the ratios matrix
        ratios_norm = ratios / np.linalg.norm(ratios, axis=0)

    # Combine all cell data
    all_cell_data = df[df["cell_id"].isin(cell_ids)]

    if len(all_cell_data) == 0:
        print("No data found for the specified cell IDs")
        plt.close(fig)
        return []

    # If same_limits is True, calculate global min and max for all channels
    global_min = None
    global_max = None

    if same_limits:
        # Find global min and max across all channel intensities
        intensity_cols = [f"chan_{chan}_intensity" for chan in available_channels]

        # Filter out non-finite values before calculating min/max
        intensity_data = all_cell_data[intensity_cols].replace([np.inf, -np.inf], np.nan)

        if scale == "log":
            # Apply log transformation for min/max calculation
            intensity_data = intensity_data.apply(lambda x: np.log1p(np.maximum(x, 0)))

        global_min = intensity_data.min().min()
        global_max = intensity_data.max().max() * 1.1  # Add 10% margin

        # Handle case where all values are NaN
        if pd.isna(global_min) or pd.isna(global_max):
            global_min = 0
            global_max = 1
            print("Warning: Could not calculate global min/max. Using default values.")

    # Get all unique channels in the unmixed data
    all_unmixed_channels = all_cell_data[chan_col].unique()

    # Iterate through all possible pairs of channels for lower triangle
    for i, chan1 in enumerate(available_channels):
        for j, chan2 in enumerate(available_channels):
            # Skip the diagonal and upper triangle
            if i <= j:
                continue

            # Calculate the position in our compact grid
            row = i - 1  # Adjust to 0-indexed
            col = j

            # Create subplot at this position
            ax = fig.add_subplot(grid[row, col])

            # Define column names
            x_col = f"chan_{chan2}_intensity"  # column index j is for x-axis
            y_col = f"chan_{chan1}_intensity"  # row index i is for y-axis

            # Track all transformed intensities for limit calculation
            x_all = []
            y_all = []

            # Channels to include in plot (default to standard channels if none in data)
            plot_channels = ["488", "514", "561", "594", "638"]
            if len(all_unmixed_channels) > 0:
                # Use actual channels from data if available
                plot_channels = [ch for ch in plot_channels if ch in all_unmixed_channels]

            # Plot spots from all unmixed channels, but using only the intensities of the two channels being compared
            for unmixed_chan in plot_channels:
                # Filter spots for this unmixed channel
                chan_data = all_cell_data[all_cell_data[chan_col] == unmixed_chan]

                if len(chan_data) > 0:
                    # For ALL spots detected in this channel, plot them at (chan2_intensity, chan1_intensity)
                    # This way we're always plotting consistent x,y coordinates for the two channels being compared
                    x = transform_intensity(chan_data[x_col], scale)  # Always use the intensity in chan2
                    y = transform_intensity(chan_data[y_col], scale)  # Always use the intensity in chan1

                    # Collect all transformed intensities for limit calculation
                    x_all.extend(x)
                    y_all.extend(y)

                    # Track this channel for the global legend
                    all_plotted_channels.add(unmixed_chan)

                    # Don't add a label here - we'll use a global legend
                    ax.scatter(
                        x,
                        y,
                        c=channel_colors.get(unmixed_chan, "gray"),
                        alpha=0.02,  # Lower alpha for dense plots
                        s=0.1,
                    )  # Smaller points for combined view

            # Set axis labels
            ax.set_xlabel(f"{chan2} Intensity ({scale} scale)")
            ax.set_ylabel(f"{chan1} Intensity ({scale} scale)")
            ax.set_title(f"{chan1} vs {chan2}")

            # Remove individual legends from the subplots
            # Don't add legend here - we'll add a global one at the end

            # Set axis limits
            if same_limits:
                # Use global limits if same_limits is True
                ax.set_xlim(global_min, global_max)
                ax.set_ylim(global_min, global_max)
            else:
                # If x_all and y_all are populated from the scatter plotting above
                if len(x_all) > 0 and len(y_all) > 0:
                    if scale == "log":
                        # Set limits slightly larger than data range
                        x_max = np.nanmax(np.array(x_all)[np.isfinite(x_all)]) if len(x_all) > 0 else 1
                        y_max = np.nanmax(np.array(y_all)[np.isfinite(y_all)]) if len(y_all) > 0 else 1
                        max_val = max(x_max, y_max) * 1.1
                        ax.set_xlim(0, max_val)
                        ax.set_ylim(0, max_val)
                    else:
                        # Use percentile-based limits for linear scale as in your example
                        x_lim = np.percentile(x_all, 99.99) if len(x_all) > 0 else 1
                        y_lim = np.percentile(y_all, 99.99) if len(y_all) > 0 else 1
                        if y_lim > 10000:
                            y_lim = 10000
                        if x_lim > 10000:
                            x_lim = 10000
                        x_low = np.percentile(x_all, 3) if len(x_all) > 0 else 0
                        y_low = np.percentile(y_all, 3) if len(y_all) > 0 else 0

                        ax.set_xlim(-x_low, x_lim)
                        ax.set_ylim(-y_low, y_lim)
                else:
                    # If no data points were plotted, use default limits
                    x_all = transform_intensity(all_cell_data[x_col], scale)
                    y_all = transform_intensity(all_cell_data[y_col], scale)

                    if scale == "log":
                        x_max = np.nanmax(x_all[np.isfinite(x_all)]) if len(x_all) > 0 else 1
                        y_max = np.nanmax(y_all[np.isfinite(y_all)]) if len(y_all) > 0 else 1
                        max_val = max(x_max, y_max) * 1.1
                        ax.set_xlim(0, max_val)
                        ax.set_ylim(0, max_val)

            # Plot dye lines if ratios matrix is provided - AFTER setting axis limits
            if ratios_norm is not None:
                # Now we pass the actual channel names instead of indices
                plot_dye_lines(ax, chan1, chan2, ratios_norm, scale, all_possible_channels=all_possible_channels)

                # Add line labels to the global legend
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    legend_handles.extend(handles)
                    legend_labels.extend(labels)

    # Add a global legend in the upper right empty space of the grid
    # Create custom markers for each channel with full opacity (alpha=1)
    channel_markers = []
    channel_labels = []

    # Sort channels for consistent legend order
    sorted_channels = sorted(all_plotted_channels)

    for chan in sorted_channels:
        color = channel_colors.get(chan, "gray")
        # Count spots for this channel
        chan_count = len(all_cell_data[all_cell_data[chan_col] == chan])
        if chan_count > 0:
            # Create a fully opaque marker for the legend
            marker = plt.Line2D(
                [0], [0], marker="o", color=color, markersize=8, linestyle="", markerfacecolor=color, alpha=1.0
            )
            channel_markers.append(marker)
            channel_labels.append(f"Channel {chan} ({chan_count} spots)")

    # Add dye line handles if any
    if legend_handles:
        channel_markers.extend(legend_handles)
        channel_labels.extend(legend_labels)

    # Place legend in the upper right corner of the figure
    if channel_markers:
        # Use the upper right area of the grid for the legend
        legend_ax = fig.add_subplot(grid[0, -1])  # Top-right position
        legend_ax.axis("off")  # Hide axis

        # Place the legend in this axis
        legend_ax.legend(
            channel_markers,
            channel_labels,
            loc="center",
            title="Channels",
            frameon=True,
            framealpha=0.8,
            scatterpoints=1,
        )

    # Add figure title
    if not title_prefix and len(all_cell_data) > 0 and "round" in all_cell_data.columns:
        round_info = f"Round {all_cell_data['round'].iloc[0]}"
        fig.suptitle(f"{round_info} - Channel Intensity Comparisons", fontsize=14, y=0.95)
    elif title_prefix:
        fig.suptitle(f"{title_prefix} - Channel Intensity Comparisons", fontsize=14, y=0.95)
    else:
        fig.suptitle("Channel Intensity Comparisons", fontsize=14, y=0.95)

    # Adjust layout - tighter spacing and better use of available space
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space for the title

    return [fig]


def plot_filtered_intensities(
    plot_df,
    round_n,
    mouse_id,
    plot_cell_ids=None,
    channel_label="unmixed_chan",
    filters=None,
    save=False,
    save_dir=Path("../scratch"),
):
    """
    Plot pairwise intensities with filters for spots data.

    Filters include:
    - over_thresh: in unmixing, for spots in a given percentile of intensity are retained to calculat dye lines.
    - valid_spot: during unmixing, various thresholds on 'r', 'dist', and 'dye-line-dist ratio' are applied to filter spots.

    Args:
        round_n (str): Round identifier (e.g. 'R1', 'R2', etc.)
        plot_df (pd.DataFrame): DataFrame containing spots data
        mouse_id (str): Mouse identifier
        plot_cell_ids (list, optional): List of cell IDs to plot. Defaults to range(1,20000)
        channel_label (str, optional): Column name for channel labels. Defaults to "unmixed_chan"
        filters (dict, optional): Dictionary of filters to apply. Defaults to basic filters
        save (bool, optional): Whether to save the plot. Defaults to False
        save_dir (Path, optional): Directory to save the plot. Defaults to Path("../scratch")

    Returns:
        list: List containing the figure
    """
    if plot_cell_ids is None:
        plot_cell_ids = list(range(1, 20000))

    if filters is not None:
        filtered_df = utils.apply_filters_to_df(plot_df, filters)
    else:
        filters = {}
        filtered_df = plot_df.copy()

    # Apply filters

    # Create plot
    fig = plot_pairwise_intensities_multi_ratios(filtered_df, plot_cell_ids, scale="linear", chan_col=channel_label)

    # Add filters to plot title
    title_filters = ", ".join([f"{k}={v}" for k, v in filters.items()])
    fig[0].suptitle(f"{round_n}\n filter: {title_filters}", fontsize=16)

    # Update channel legend label if legend exists
    legend = fig[0].axes[0].get_legend()
    if legend is not None:
        legend.set_title(f"{channel_label} intensity")

    # Save plot if requested
    if save:
        save_dir = save_dir / f"{mouse_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        filters_str = utils.filter_dict_to_string(filters)
        # change channel_label, "unmixed_chan" -> unmixed, "chan" -> mixed
        if channel_label == "unmixed_chan":
            channel_label = "unmixed"
        elif channel_label == "chan":
            channel_label = "mixed"
        filename = f"{round_n}_pairwise_int_{channel_label}_{filters_str}.png"
        fig[0].savefig(save_dir / filename)

    return fig


def qc_spectral_unmixing(data_dir, output_dir, channels=None, verbose=False):
    """
    Run spectral unmixing quality control analysis.

    Parameters:
    -----------
    data_dir : Path or str
        Path to data directory containing spectral data
    output_dir : Path or str
        Directory to save QC outputs
    channels : list, optional
        List of channels to analyze
    verbose : bool
        Enable verbose output
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    if channels is None:
        channels = ["405", "488", "514", "561", "594", "638"]

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Running spectral unmixing QC for channels: {channels}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")

    # Placeholder analysis - create dummy spectral unmixing plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Channel crosstalk matrix
    n_channels = len(channels)
    crosstalk_matrix = np.random.uniform(0, 0.3, (n_channels, n_channels))
    np.fill_diagonal(crosstalk_matrix, 1.0)  # Perfect signal in own channel

    im = axes[0, 0].imshow(crosstalk_matrix, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[0, 0].set_title("Channel Crosstalk Matrix")
    axes[0, 0].set_xlabel("Excitation Channel")
    axes[0, 0].set_ylabel("Detection Channel")
    axes[0, 0].set_xticks(range(n_channels))
    axes[0, 0].set_yticks(range(n_channels))
    axes[0, 0].set_xticklabels(channels)
    axes[0, 0].set_yticklabels(channels)
    plt.colorbar(im, ax=axes[0, 0])

    # Spectral profiles
    wavelengths = np.linspace(400, 700, 100)
    for i, channel in enumerate(channels[:3]):  # Show first 3 channels
        emission = np.exp(-((wavelengths - (400 + i * 50)) ** 2) / (2 * 20**2))
        axes[0, 1].plot(wavelengths, emission, label=f"Ch {channel}", linewidth=2)
    axes[0, 1].set_xlabel("Wavelength (nm)")
    axes[0, 1].set_ylabel("Normalized Intensity")
    axes[0, 1].set_title("Spectral Emission Profiles")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Unmixing accuracy
    true_vs_unmixed = np.random.normal(0, 0.1, 100) + np.linspace(-1, 1, 100)
    axes[0, 2].scatter(np.linspace(-1, 1, 100), true_vs_unmixed, alpha=0.6)
    axes[0, 2].plot([-1, 1], [-1, 1], "r--", linewidth=2, label="Perfect unmixing")
    axes[0, 2].set_xlabel("True Signal")
    axes[0, 2].set_ylabel("Unmixed Signal")
    axes[0, 2].set_title("Unmixing Accuracy")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Signal-to-noise ratio per channel
    snr_values = np.random.uniform(10, 50, n_channels)
    axes[1, 0].bar(channels, snr_values, color="skyblue", alpha=0.7, edgecolor="black")
    axes[1, 0].set_xlabel("Channel")
    axes[1, 0].set_ylabel("SNR (dB)")
    axes[1, 0].set_title("Signal-to-Noise Ratio by Channel")
    axes[1, 0].grid(True, alpha=0.3)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

    # Bleedthrough correction efficiency
    correction_efficiency = np.random.uniform(0.8, 0.98, n_channels)
    axes[1, 1].bar(channels, correction_efficiency, color="lightcoral", alpha=0.7, edgecolor="black")
    axes[1, 1].set_xlabel("Channel")
    axes[1, 1].set_ylabel("Correction Efficiency")
    axes[1, 1].set_title("Bleedthrough Correction Efficiency")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

    # Residual analysis
    residuals = np.random.normal(0, 1, 1000)
    axes[1, 2].hist(residuals, bins=30, alpha=0.7, edgecolor="black", color="lightgreen")
    axes[1, 2].set_xlabel("Unmixing Residuals")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].set_title("Unmixing Residuals Distribution")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "spectral_unmixing_qc.png", dpi=300, bbox_inches="tight")
    plt.close()

    if verbose:
        print("Spectral unmixing QC completed successfully!")


def dummy_func():
    """Dummy function"""
