import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.gridspec as gridspec
# -------------------------------------------------------------------------------------------------
# Data handling functions
# -------------------------------------------------------------------------------------------------


def load_mean_cxg(dataset, mean_csv_dict, value="mean"):
    """
    Load mean cell x gene data from the dataset.

    Parameters
    ----------
    dataset : HCRDataset
        The HCR dataset containing round information
    mean_csv_dict : dict, optional
        Dictionary mapping round keys to CSV file paths containing mean expression data.
        If None, uses default file paths.
        Example:
        {
            "R1": "/path/to/mean_data_R1.csv",
            "R2": "/path/to/mean_data_R2.csv"
        }
    value : str, optional
        Value for cell x gene matrix, may be "mean", "mean_bg_corr", or "sum", or other metric

    Returns
    -------
    pandas.DataFrame
        Cell x gene matrix with mean intensity values, with cell_id as index and genes as columns
    """
    if not mean_csv_dict:
        raise ValueError("mean_csv_dict must be provided with round keys and CSV file paths.")

    dfs = []
    for round_key in mean_csv_dict.keys():
        pm = dataset.rounds[round_key].processing_manifest
        mean_df = pd.read_csv(mean_csv_dict[round_key])

        # subtract background
        mean_df["mean_bg_corr"] = mean_df["mean"] - mean_df["background"]

        mean_df["gene"] = (
            mean_df["channel"].astype(str).map(pm["gene_dict"]).apply(lambda x: x["gene"] if isinstance(x, dict) else x)
        )
        mean_df["round"] = round_key
        dfs.append(mean_df)

    # has: channel	cell_id	sum	count	mean	gene	round
    dfs = pd.concat(dfs, axis=0)
    gene_order = dfs["gene"].unique().tolist()

    mean_cxg = dfs.copy()
    # put cell id as index and gene as columns, with the specified value as values
    mean_cxg = mean_cxg.pivot_table(index="cell_id", columns="gene", values=value).reset_index()
    # sort cols by gene order
    mean_cxg = mean_cxg.reindex(columns=["cell_id"] + gene_order)

    mean_cxg = mean_cxg.set_index("cell_id")
    mean_cxg.dropna(inplace=True)

    print(f"Loaded mean cell x gene matrix with shape: {mean_cxg.shape}")
    print(f"Genes: {mean_cxg.columns.tolist()}")

    return mean_cxg


def spot_count_cell_x_gene_coreg(dataset, 
                                 coreg_spots, 
                                 ophys_mfish_match_df,
                                 save_coreg_spots=False,
                                 output_dir=None,
                                 filtered_spots=True,
                                 r=0.3,
                                 dist=1):
    """
    Create a cell x gene table from the coregistered spots.
    Used for July 2025 data club, may refactor later.

    Parameters:
    - dataset: The HCR dataset containing the rounds and mixed spots.
    - coreg_spots_filtered: DataFrame containing filtered coregistered mixed spots.
    Returns:
    - DataFrame containing cell x gene counts.
    """
    mouse_id = dataset.mouse_id
    if filtered_spots:
        coreg_spots_filtered = coreg_spots[(coreg_spots['r'] > r) & (coreg_spots['dist'] < dist)]
        print(f"Number of coregistered mixed spots after filtering: {len(coreg_spots_filtered)}")
    else:
        coreg_spots_filtered = coreg_spots
        print(f"Number of coregistered mixed spots without filtering: {len(coreg_spots_filtered)}")
    spot_counts = coreg_spots_filtered.groupby(['round','chan','cell_id']).size().reset_index(name='spot_count')
    ch_gene_table = dataset.create_channel_gene_table(spots_only=True)
    spot_counts = spot_counts.merge(ch_gene_table, left_on=['round', 'chan'], 
                                    right_on = ["Round", "Channel"], how='left')
    spot_counts = spot_counts.drop(columns=['Round', 'Channel'])
    spot_counts = spot_counts.rename(columns={'Gene': 'gene'})


    gene_order = spot_counts['gene'].unique()
    spot_counts_pivot = spot_counts.pivot(index='cell_id', 
                                        columns='gene', values='spot_count',
                                        ).fillna(0)
    spot_counts_pivot = spot_counts_pivot.reindex(columns=gene_order, fill_value=0)


    # save the spot counts to a csv file
    spot_counts_merged= spot_counts.merge(ophys_mfish_match_df, left_on='cell_id', right_on='ls_id', how='left')
    if save_coreg_spots and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        spot_counts_merged.to_csv(Path(output_dir / f'{mouse_id}_cxg_mixed_spot_counts.csv'), index=False)
    return spot_counts, spot_counts_pivot, spot_counts_merged


# -------------------------------------------------------------------------------------------------
# Cell x gene plotting functions
# -------------------------------------------------------------------------------------------------


def plot_cell_x_gene_simple(cxg, clip_range=(0, 50), sort_gene=None, fig_size=(4, 6)):
    """
    Plot the cell x gene matrix as an image with inverted colormap.

    Parameters
    ----------
    cxg : pd.DataFrame
        Cell x gene matrix with genes as columns and cells as rows.
    clip_range : tuple
        Range to clip the values in the cell x gene matrix.
    sort_gene : str, optional
        Gene to sort the cell x gene matrix by. If None, sorts by the first gene.
        Default is None.
    fig_size : tuple
        Size of the figure to plot.

    """
    if not isinstance(cxg, pd.DataFrame):
        raise ValueError("Input cxg must be a pandas DataFrame.")

    cxg = cxg.copy()  # avoid modifying the original DataFrame
    # set color min/max
    cxg = cxg.fillna(0)  # fill NaN values with 0

    # make int
    cxg = cxg.astype(int)
    cxg = cxg.clip(lower=clip_range[0], upper=clip_range[1])

    # sort by gene if specified
    if sort_gene is not None and sort_gene not in cxg.columns:
        raise ValueError(f"Gene '{sort_gene}' not found in cell x gene matrix columns.")
    if sort_gene is not None:
        cxg = cxg.sort_values(by=sort_gene, ascending=False)

    fig, ax = plt.subplots(figsize=fig_size)

    ax.imshow(
        cxg,
        aspect="auto",
        cmap="gray_r",
        interpolation="none",
    )
    # show colorbar
    cbar = plt.colorbar(ax.imshow(cxg, aspect="auto", cmap="gray_r", interpolation="none"), ax=ax)
    cbar.set_label("Gene Expression Count", rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)
    # cbar.set_clim(clip_range[0], clip_range[1])

    # add gene names from dataframe
    # plt.yticks(ticks=range(len(cxg.index)), labels=cxg.index)
    ax.set_xticks(ticks=range(len(cxg.columns)), labels=cxg.columns, rotation=90)

    # colorbar

    return fig


def plot_cell_x_gene_clustered(
    cxg,
    clip_range=(0, 50),
    sort_gene=None,
    fig_size=(4, 6),
    k=3,
    add_cluster_labels=True,
    cbar_label="Gene Expression Count",
):
    """
    Plot the cell x gene matrix as an image with inverted colormap and K-means clustering.

    Parameters
    ----------
    cxg : pd.DataFrame
        Cell x gene matrix with genes as columns and cells as rows.
    clip_range : tuple
        Range to clip the values in the cell x gene matrix.
    sort_gene : str, optional
        Gene to sort the cell x gene matrix by. If None, performs clustering instead.
        Default is None.
    fig_size : tuple
        Size of the figure to plot.
    k : int
        Number of clusters for K-means clustering. Default is 3.
    add_cluster_labels : bool
        Whether to add green dashed lines and labels to indicate cluster boundaries.
        Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    cluster_labels : np.ndarray
        Array of cluster assignments for each cell.
    sorted_cell_ids : pd.Index
        Index of cell IDs in the same sorted order as cluster_labels.
    """
    if not isinstance(cxg, pd.DataFrame):
        raise ValueError("Input cxg must be a pandas DataFrame.")

    cxg = cxg.copy()  # avoid modifying the original DataFrame
    # set color min/max
    cxg = cxg.fillna(0)  # fill NaN values with 0

    # make int
    cxg = cxg.astype(int)
    cxg = cxg.clip(lower=clip_range[0], upper=clip_range[1])

    # Perform clustering or sorting
    if sort_gene is not None and sort_gene in cxg.columns:
        # Sort by specified gene
        cxg = cxg.sort_values(by=sort_gene, ascending=False)
        cluster_labels = None
        sorted_cell_ids = cxg.index  # Store the sorted cell IDs
    else:
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cxg.values)

        # Sort by cluster labels first, then by total expression within each cluster
        cxg["cluster"] = cluster_labels
        cxg["total_expression"] = cxg.drop("cluster", axis=1).sum(axis=1)
        cxg = cxg.sort_values(["cluster", "total_expression"], ascending=[True, False])

        # Update cluster labels to match the sorted order
        cluster_labels = cxg["cluster"].values

        # Store the sorted cell IDs before dropping helper columns
        sorted_cell_ids = cxg.index

        # Remove helper columns
        cxg = cxg.drop(["cluster", "total_expression"], axis=1)

    fig, ax = plt.subplots(figsize=fig_size)

    # Plot the heatmap
    im = ax.imshow(cxg, aspect="auto", cmap="gray_r", interpolation="none")

    # show colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)

    # add gene names from dataframe
    ax.set_xticks(ticks=range(len(cxg.columns)), labels=cxg.columns, rotation=90)

    # Add cluster labels if clustering was performed and requested
    if cluster_labels is not None and add_cluster_labels:
        # Find cluster boundaries
        cluster_changes = np.where(np.diff(cluster_labels) != 0)[0] + 0.5

        # Add horizontal dashed lines at cluster boundaries
        for boundary in cluster_changes:
            ax.axhline(y=boundary, color="green", linestyle="--", linewidth=2.0, alpha=0.9)

        # Add cluster labels
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            # Find the middle position of each cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_center = (cluster_indices[0] + cluster_indices[-1]) / 2

            # Add text label
            ax.text(
                -0.5,
                cluster_center,
                f"C{cluster_id}",
                color="green",
                fontweight="bold",
                fontsize=14,
                verticalalignment="center",
                horizontalalignment="right",
            )

    return fig, cluster_labels, sorted_cell_ids


# -------------------------------------------------------------------------------------------------
# Cell mean centroid functions
# Note: These functions are used to visualize cell centroids with histograms of mean fluorescence,
#       Not cell x gene per se, but need gene call information to plot meaningful clusters.
# -------------------------------------------------------------------------------------------------
def cell_mean_centroid_df(dataset, mean_csv, round_key):
    """
    Load cell mean flouresence data from a CSV file and merges with cell information.
    """

    # filter volume with 5 - 95 percentile
    cell_info = dataset.rounds[
        round_key
    ].get_cell_info()  # Double check if the right cells..., since R1 is default in loader

    # filter volume
    cell_info = cell_info[
        (cell_info["volume"] > cell_info["volume"].quantile(0.05))
        & (cell_info["volume"] < cell_info["volume"].quantile(0.95))
    ]
    filt_cells = cell_info.cell_id.values

    mean_df = pd.read_csv(mean_csv)
    # merge with cell_info
    mean_df = mean_df.merge(cell_info, on="cell_id", how="left")

    # filter by cell_id
    mean_df = mean_df[mean_df["cell_id"].isin(filt_cells)]

    print(f"Filtered mean_df shape: {mean_df.shape}")
    return mean_df


def plot_centroids_with_hist(
    df,
    orientation="XY",
    n_samples=None,
    color_col=None,
    cmap="viridis",
    clip_range=(None, None),
    xlims=(None, None),
    ylims=(None, None),
    show_colorbar=True,
    title_str=None,
    random_state=42,
    fig_size=(12, 8),
    save: bool = False,
    output_dir: str = None,
    filename_str: str = None,
):
    """
    Plot cell centroids with a vertical KDE histogram of the color column.

    TODO: maybe can move out of cellxgene?

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing centroid coordinates
    orientation : str
        One of 'XY', 'ZX', or 'ZY'
    n_samples : int, optional
        Number of random cells to plot
    color_col : str, optional
        Column to use for coloring and histogram
    cmap : str
        Colormap for scatter plot
    clip_range : tuple
        Range to clip the color column values
    random_state : int
        Random seed for reproducibility
    fig_size : tuple
        Figure size (width, height)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    """
    # Sample random cells if specified
    if n_samples is not None and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=random_state)

    # Clip color column if specified
    if color_col is not None:
        if color_col not in df.columns:
            raise ValueError(f"Color column '{color_col}' not found in DataFrame.")
        df = df.copy()
        df[color_col] = df[color_col].clip(lower=clip_range[0], upper=clip_range[1])

    # Define coordinate mappings
    coords = {
        "XY": ("x_centroid", "y_centroid", "XY Plane"),
        "ZX": ("x_centroid", "z_centroid", "ZX Plane"),
        "ZY": ("y_centroid", "z_centroid", "ZY Plane"),
    }

    if orientation not in coords:
        raise ValueError("Orientation must be one of: 'XY', 'ZX', 'ZY'")

    x_coord, y_coord, plane = coords[orientation]

    # Create figure with subplots
    #fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=fig_size, gridspec_kw={"width_ratios": [3, 1]})

    fig = plt.figure(figsize=(fig_size))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.2)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])  # Share both axes

    # --- Scatter plot ---
    if color_col is not None:
        scatter = ax_scatter.scatter(df[x_coord], df[y_coord], alpha=0.6, c=df[color_col], cmap=cmap, s=8)
        # Add colorbar
        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax_scatter)
            cbar.set_label(color_col, rotation=270, labelpad=15)
    else:
        ax_scatter.scatter(df[x_coord], df[y_coord], alpha=0.6, s=8)

    # Set scatter plot properties
    if xlims[0] is not None or xlims[1] is not None:
        ax_scatter.set_xlim(xlims[0], xlims[1])
    else:
        ax_scatter.set_xlim(df[x_coord].min() - 10, df[x_coord].max() + 10)

    if ylims[0] is not None or ylims[1] is not None:
        ax_scatter.set_ylim(ylims[0], ylims[1])
    else:
        ax_scatter.set_ylim(df[y_coord].min() - 10, df[y_coord].max() + 10)

    #ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xlabel(f"{x_coord}")
    ax_scatter.set_ylabel(f"{y_coord}")
    ax_scatter.set_title(f"{plane}")

    # Reverse y-axis for consistency
    if orientation in ["ZX", "ZY", "XY"]:
        ax_scatter.invert_yaxis()

    # --- Histogram/KDE plot ---
    if color_col is not None:
        # Get data for histogram - we want to show color values distributed across y-axis coordinates
        valid_data = df[[y_coord, color_col]].dropna()

        if len(valid_data) > 0:
            # Create binned averages of color values across y-coordinate bins
            y_bins = np.linspace(valid_data[y_coord].min(), valid_data[y_coord].max(), 30)
            y_centers = (y_bins[:-1] + y_bins[1:]) / 2

            # Calculate average color value in each y-coordinate bin
            binned_means = []
            for i in range(len(y_bins) - 1):
                mask = (valid_data[y_coord] >= y_bins[i]) & (valid_data[y_coord] < y_bins[i + 1])
                if mask.sum() > 0:
                    binned_means.append(valid_data[color_col][mask].sum())

                else:
                    binned_means.append(0)
            hist_color = plt.cm.get_cmap(cmap)(0.8)
            # Plot as horizontal bars showing average color value at each y position
            ax_hist.barh(
                y_centers,
                binned_means,
                height=(y_bins[1] - y_bins[0]) * 0.8,
                alpha=0.7,
                color=hist_color,
                edgecolor="black",
                linewidth=0.5,
            )

            # Smooth the data
            # set color to top of cmap

            smoothed_means = gaussian_filter1d(binned_means, sigma=1.5)
            ax_hist.plot(smoothed_means, y_centers, color=hist_color, linewidth=2, alpha=0.8)

            ax_hist.set_ylabel(f"{y_coord}")
            # ax_hist.set_xlabel(f'Summed {color_col} intensity')
            ax_hist.set_xlabel("Count")
            # ax_hist.set_title(f'{color_col} intensity vs {y_coord}')

            # Set y-limits to match the scatter plot
            ax_hist.set_ylim(ax_scatter.get_ylim())

            # Set x-limits for color values
            # if clip_range[0] is not None or clip_range[1] is not None:
            #     x_min = clip_range[0] if clip_range[0] is not None else valid_data[color_col].min()
            #     x_max = clip_range[1] if clip_range[1] is not None else valid_data[color_col].max()
            #     ax_hist.set_xlim(0, x_max)
            #     print(f"Histogram x-limits set to: {x_min} - {x_max}")

            # ylabels off
            ax_hist.set_yticks([])
            # ylabel off
            ax_hist.set_ylabel("")

            if title_str is not None:
                ax_hist.set_title(title_str)

        else:
            ax_hist.text(
                0.5, 0.5, "No valid data\nfor histogram", ha="center", va="center", transform=ax_hist.transAxes
            )
            ax_hist.set_title("No Data")
    else:
        # Hide histogram if no color column
        ax_hist.text(0.5, 0.5, "No color column\nspecified", ha="center", va="center", transform=ax_hist.transAxes)
        ax_hist.set_title("No Color Data")

    # Add sample size information
    sample_text = f"n={len(df)}"
    # add text to the top right corner of the scatter plot
    ax_hist.text(
        0.95,
        0.95,
        sample_text,
        transform=ax_hist.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"),
    )

    if save and output_dir is not None:
        filename_str = "" if filename_str is None else "_" + filename_str
        output_path = f"{output_dir}/centroids_{orientation}{filename_str}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {output_path}")

    return fig


def calculate_cluster_percentages(cluster_labels):
    """
    Calculate the percentage of each cluster size relative to the total number of cells.
    
    Parameters:
    cluster_labels (array-like): Array of cluster labels for each cell.
    
    Returns:
    pd.DataFrame: DataFrame containing cluster sizes and their percentages.
    """
    cluster_sizes = np.bincount(cluster_labels)
    total_cells = len(cluster_labels)
    cluster_percentages = cluster_sizes / total_cells * 100
    cluster_df = pd.DataFrame({
        'Cluster': np.arange(len(cluster_sizes)),
        'Size': cluster_sizes,
        'Percentage': cluster_percentages
    })
    return cluster_df


def plot_cluster_centroids(cell_info_clusters, cluster_n, save=False):
    """
    Plot the centroids of a specific cluster.
    
    Parameters:
    cluster_n (int): The cluster number to plot.
    """
    # Filter the cell_info_clusters DataFrame for the specified cluster
    plot_cluster_df = cell_info_clusters[cell_info_clusters['cluster_label'] == cluster_n]
    
    # Set the cluster label to 1 for plotting purposes
    plot_cluster_df[:, 'cluster_label'] = 1
    
    # Get the maximum x and y coordinates for setting limits
    max_x = cell_info_clusters['x_centroid'].max()
    max_y = cell_info_clusters['y_centroid'].max()
    
    # Plot the centroids with histogram
    fig = plot_centroids_with_hist(plot_cluster_df,
                                  orientation="XY",
                                  color_col="cluster_label",
                                  cmap="Greys_r",
                                  xlims=(0, max_x),
                                  ylims=(0, max_y),
                                  show_colorbar=False,
                                  fig_size=(4, 4),
                                  save=save,
                                  output_dir=Path("/root/capsule/scratch/"),
                                  title_str=f"Cluster {cluster_n}",
                                  )
    
    # if not save:
    #     #plt.show()

    return fig