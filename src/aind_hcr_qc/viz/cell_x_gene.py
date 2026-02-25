from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from aind_hcr_qc.utils.utils import saveable_plot
import aind_hcr_data_loader.filters as hcr_filters


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


def spot_count_cell_x_gene_coreg(
    dataset,
    coreg_spots,
    ophys_mfish_match_df,
    save_coreg_spots=False,
    output_dir=None,
    filtered_spots=True,
    r=0.3,
    dist=1,
):
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
        coreg_spots_filtered = coreg_spots[(coreg_spots["r"] > r) & (coreg_spots["dist"] < dist)]
        print(f"Number of coregistered mixed spots after filtering: {len(coreg_spots_filtered)}")
    else:
        coreg_spots_filtered = coreg_spots
        print(f"Number of coregistered mixed spots without filtering: {len(coreg_spots_filtered)}")
    spot_counts = coreg_spots_filtered.groupby(["round", "chan", "cell_id"]).size().reset_index(name="spot_count")
    ch_gene_table = dataset.create_channel_gene_table(spots_only=True)
    spot_counts = spot_counts.merge(ch_gene_table, left_on=["round", "chan"], right_on=["Round", "Channel"], how="left")
    spot_counts = spot_counts.drop(columns=["Round", "Channel"])
    spot_counts = spot_counts.rename(columns={"Gene": "gene"})

    gene_order = spot_counts["gene"].unique()
    spot_counts_pivot = spot_counts.pivot(
        index="cell_id",
        columns="gene",
        values="spot_count",
    ).fillna(0)
    spot_counts_pivot = spot_counts_pivot.reindex(columns=gene_order, fill_value=0)

    # save the spot counts to a csv file
    spot_counts_merged = spot_counts.merge(ophys_mfish_match_df, left_on="cell_id", right_on="ls_id", how="left")
    if save_coreg_spots and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        spot_counts_merged.to_csv(Path(output_dir / f"{mouse_id}_cxg_mixed_spot_counts.csv"), index=False)
    return spot_counts, spot_counts_pivot, spot_counts_merged


# -------------------------------------------------------------------------------------------------
# Cell x gene plotting functions
# -------------------------------------------------------------------------------------------------


def _build_gene_labels(gene_columns: list, gene_label: str, dataset) -> list:
    """
    Build x-axis tick labels for a cell x gene plot.

    Parameters
    ----------
    gene_columns : list of str
        Ordered gene names (column names of the cxg pivot table).
    gene_label : {'gene', 'round_channel_gene'}
        'gene'               → return gene_columns unchanged.
        'round_channel_gene' → map each gene to its "R{n}_ch{ch}_{gene}" label
                               using the dataset's channel-gene table.
    dataset : HCRDataset or None
        Required when gene_label == 'round_channel_gene'.

    Returns
    -------
    list of str
    """
    if gene_label == 'gene' or dataset is None:
        return gene_columns

    channel_gene_table = dataset.create_channel_gene_table()
    if 'round_channel_gene' not in channel_gene_table.columns:
        # Fallback: build it on the fly if the dataset version is old
        channel_gene_table['round_channel_gene'] = (
            channel_gene_table['Round'] + '_ch' + channel_gene_table['Channel'].astype(str)
            + '_' + channel_gene_table['Gene']
        )

    gene_to_label = dict(zip(channel_gene_table['Gene'], channel_gene_table['round_channel_gene']))
    return [gene_to_label.get(g, g) for g in gene_columns]


def plot_cell_x_gene_simple(cxg, clip_range=(0, 50), sort_gene=None, fig_size=(4, 6), ax=None, title=None, 
                           gene_sort='alphabetical', dataset=None, cbar_label="Transcript count",
                           gene_label='gene'):
    """
    Plot the cell x gene matrix as an image with inverted colormap.

    Parameters
    ----------
    cxg : pd.DataFrame
        Cell x gene matrix with genes as columns and cells as rows.
    clip_range : tuple
        Range to clip the values in the cell x gene matrix.
    sort_gene : str, optional
        DEPRECATED: Use gene_sort parameter instead. Gene to sort cells by.
    fig_size : tuple
        Size of the figure to plot.
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure.
    title : str, optional
        Title for the plot.
    gene_sort : str, optional
        How to sort genes (columns). Options:
        - 'alphabetical' (default): Sort genes A-Z
        - 'round_channel': Sort by imaging round then channel (requires dataset parameter)
        - gene name string: Sort cells by expression of that gene
    dataset : HCRDataset, optional
        Required if gene_sort='round_channel'. Used to get channel-gene mapping.
    gene_label : {'gene', 'round_channel_gene'}, optional
        Column from the channel-gene table to use as x-axis tick labels.
        'gene' (default) shows only the gene name; 'round_channel_gene' shows
        e.g. "R2_ch647_Gad2". Requires ``dataset`` when not 'gene'.
    """
    if not isinstance(cxg, pd.DataFrame):
        raise ValueError("Input cxg must be a pandas DataFrame.")

    cxg = cxg.copy()  # avoid modifying the original DataFrame
    # set color min/max
    cxg = cxg.fillna(0)  # fill NaN values with 0

    # make int
    cxg = cxg.astype(int)
    cxg = cxg.clip(lower=clip_range[0], upper=clip_range[1])

    # Handle gene column sorting
    if gene_sort == 'round_channel':
        if dataset is None:
            raise ValueError("dataset parameter required when gene_sort='round_channel'")
        channel_gene_table = dataset.create_channel_gene_table()
        gene_order = channel_gene_table.sort_values(['Round', 'Channel'])['Gene'].tolist()
        gene_order = [g for g in gene_order if g in cxg.columns]
        cxg = cxg[gene_order]
    elif gene_sort == 'alphabetical':
        cxg = cxg[sorted(cxg.columns)]
    elif gene_sort in cxg.columns:
        # gene_sort is a gene name, sort cells by it
        cxg = cxg.sort_values(by=gene_sort, ascending=False)
    elif gene_sort is not None and gene_sort != 'alphabetical':
        raise ValueError(f"gene_sort '{gene_sort}' not recognized. Use 'alphabetical', 'round_channel', or a gene name.")
    
    # Legacy support for sort_gene parameter
    if sort_gene is not None:
        if sort_gene not in cxg.columns:
            raise ValueError(f"Gene '{sort_gene}' not found in cell x gene matrix columns.")
        cxg = cxg.sort_values(by=sort_gene, ascending=False)

    # Build x-axis tick labels
    x_labels = _build_gene_labels(cxg.columns.tolist(), gene_label, dataset)

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    ax.imshow(
        cxg,
        aspect="auto",
        cmap="gray_r",
        interpolation="none",
    )
    # show colorbar
    cbar = plt.colorbar(ax.imshow(cxg, aspect="auto", cmap="gray_r", interpolation="none"), ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)
    # cbar.set_clim(clip_range[0], clip_range[1])

    # add gene names from dataframe
    # plt.yticks(ticks=range(len(cxg.index)), labels=cxg.index)
    ax.set_xticks(ticks=range(len(cxg.columns)), labels=x_labels, rotation=90)
    ax.set_title(title)
    # colorbar

    return ax


def plot_cell_x_gene_clustered(
    cxg,
    clip_range=(0, 50),
    sort_gene=None,
    fig_size=(4, 6),
    k=3,
    add_cluster_labels=True,
    cbar_label="Transcript count",
    title=None,
    ax=None,
    gene_sort='alphabetical',
    dataset=None,
    cluster_sort_gene='Gad2',
    gene_label='gene',
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
        DEPRECATED: Gene to sort cells by. If None, performs clustering instead.
    fig_size : tuple
        Size of the figure to plot.
    k : int
        Number of clusters for K-means clustering. Default is 3.
    add_cluster_labels : bool
        Whether to add green dashed lines and labels to indicate cluster boundaries.
        Default is True.
    cbar_label : str
        Label for colorbar. Default is "Transcript count".
    title : str, optional
        Title for the plot.
    ax : matplotlib axis, optional
        Axis to plot on.
    gene_sort : str, optional
        How to sort genes (columns). Options:
        - 'alphabetical' (default): Sort genes A-Z
        - 'round_channel': Sort by imaging round then channel (requires dataset parameter)
        - gene name string: Sort cells by expression of that gene
    dataset : HCRDataset, optional
        Required if gene_sort='round_channel'. Used to get channel-gene mapping.
    cluster_sort_gene : str, optional
        After KMeans, order the *clusters themselves* by their mean expression of this gene,
        ascending (lowest-expressing cluster at the top, highest at the bottom).
        Default is 'Gad2'. Falls back to total mean expression if the gene is absent.
    gene_label : {'gene', 'round_channel_gene'}, optional
        Column from the channel-gene table to use as x-axis tick labels.
        'gene' (default) shows only the gene name; 'round_channel_gene' shows
        e.g. "R2_ch647_Gad2". Requires ``dataset`` when not 'gene'.

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

    # Handle gene column sorting first (before clustering/sorting rows)
    if gene_sort == 'round_channel':
        if dataset is None:
            raise ValueError("dataset parameter required when gene_sort='round_channel'")
        channel_gene_table = dataset.create_channel_gene_table()
        gene_order = channel_gene_table.sort_values(['Round', 'Channel'])['Gene'].tolist()
        gene_order = [g for g in gene_order if g in cxg.columns]
        cxg = cxg[gene_order]
    elif gene_sort == 'alphabetical':
        cxg = cxg[sorted(cxg.columns)]
    elif gene_sort in cxg.columns:
        # gene_sort is a gene name, will be used for cell sorting below
        pass
    elif gene_sort is not None and gene_sort != 'alphabetical':
        raise ValueError(f"gene_sort '{gene_sort}' not recognized. Use 'alphabetical', 'round_channel', or a gene name.")

    # Build x-axis tick labels (after column order is finalised)
    x_labels = _build_gene_labels(cxg.columns.tolist(), gene_label, dataset)

    # Perform clustering or sorting (for rows/cells)
    # Use gene_sort if it's a gene name, otherwise use legacy sort_gene, otherwise cluster
    cell_sort_gene = gene_sort if gene_sort in cxg.columns else sort_gene
    
    if cell_sort_gene is not None and cell_sort_gene in cxg.columns:
        # Sort by specified gene
        cxg = cxg.sort_values(by=cell_sort_gene, ascending=False)
        cluster_labels = None
        sorted_cell_ids = cxg.index
        y_label = f"Cells sorted by {cell_sort_gene}"
    else:
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cxg.values)

        # --- Sort clusters by their mean expression of cluster_sort_gene ---
        # (lowest mean → top of plot, highest mean → bottom)
        cxg["_cluster"] = cluster_labels
        if cluster_sort_gene and cluster_sort_gene in cxg.columns:
            cluster_means = cxg.groupby("_cluster")[cluster_sort_gene].mean()
        else:
            # fall back to total mean expression per cluster
            cluster_means = cxg.drop(columns=["_cluster"]).assign(_cluster=cluster_labels).groupby("_cluster").mean().mean(axis=1)

        # Map original cluster id → rank (0 = lowest mean, k-1 = highest mean)
        sorted_cluster_ids = cluster_means.sort_values(ascending=True).index
        rank_map = {orig: rank for rank, orig in enumerate(sorted_cluster_ids)}
        cxg["_cluster_rank"] = cxg["_cluster"].map(rank_map)

        # Sort rows by cluster rank only — no within-cluster sorting
        cxg = cxg.sort_values("_cluster_rank", ascending=True)

        # Expose remapped cluster labels (0 = lowest gene-expressing cluster)
        cluster_labels = cxg["_cluster_rank"].values

        # Store sorted cell IDs before dropping helper columns
        sorted_cell_ids = cxg.index

        # Remove helper columns
        cxg = cxg.drop(columns=["_cluster", "_cluster_rank"])

        sort_label = cluster_sort_gene if cluster_sort_gene and cluster_sort_gene in cxg.columns else "total expr"
        y_label = f"Cells (clusters sorted by mean {sort_label} ↑)"

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    # Plot the heatmap
    im = ax.imshow(cxg, aspect="auto", cmap="gray_r", interpolation="none")

    # show colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)

    # add gene names from dataframe
    ax.set_xticks(ticks=range(len(cxg.columns)), labels=x_labels, rotation=90)

    # y-axis label
    ax.set_ylabel(y_label, fontsize=9)

    # Add cluster labels if clustering was performed and requested
    if cluster_labels is not None and add_cluster_labels:
        # Find cluster boundaries
        cluster_changes = np.where(np.diff(cluster_labels) != 0)[0] + 0.5

        # Add horizontal dashed lines at cluster boundaries
        for boundary in cluster_changes:
            ax.axhline(y=boundary, color="green", linestyle="--", linewidth=2.0, alpha=0.9)

        # Add cluster ID labels (number only, no "C" prefix)
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_center = (cluster_indices[0] + cluster_indices[-1]) / 2

            ax.text(
                -0.5,
                cluster_center,
                str(cluster_id),
                color="green",
                fontweight="normal",
                fontsize=10,
                verticalalignment="center",
                horizontalalignment="right",
            )

    if title is not None:
        ax.set_title(title)

    return fig, cluster_labels, sorted_cell_ids


from sklearn.metrics import silhouette_score


def _plot_inhibitory_gene_dist(cxg_pivot, gene, threshold, log_transform,ax=None, drop_zeros=True):
    """
    Plot the per-cell spot-count distribution for one inhibitory gene on a single axis.

    Data is always displayed on a log2(x+1) scale for readability.  The
    threshold line is positioned and annotated differently depending on
    whether the caller is working with log-transformed or raw data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    cxg_pivot : pd.DataFrame
        Cell × gene matrix (cells as rows, genes as columns).
    gene : str
        Gene name to plot (must be a column in cxg_pivot).
    threshold : float
        Inhibitory-cell threshold for this gene, in the same space as
        cxg_pivot values (i.e. already log2 if log_transform=True,
        raw counts if log_transform=False).
    log_transform : bool
        Whether cxg_pivot values are already log2(x+1) transformed.
    drop_zeros : bool
        If True (default), cells with zero expression are excluded from the
        KDE so the non-expressing peak does not dominate the plot.  A compact
        note showing the zero count and percentage is added below the
        threshold annotation.
    """
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if gene not in cxg_pivot.columns:
        ax.text(0.5, 0.5, f"{gene}\nnot found", ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
        ax.axis('off')
        return

    raw_counts = cxg_pivot[gene]
    n_total = len(raw_counts)

    # Determine zero mask in the original (possibly log) space
    n_zeros = int((raw_counts == 0).sum())
    pct_zeros = 100.0 * n_zeros / n_total if n_total > 0 else 0.0

    if drop_zeros:
        raw_counts = raw_counts[raw_counts > 0]

    if log_transform:
        # Values are already log2(x+1); plot directly
        plot_data = raw_counts
        threshold_x = threshold
        xlabel = "log2(x+1)"
    else:
        # Values are raw counts; convert to log2(x+1) for display only
        plot_data = np.log2(raw_counts + 1)
        threshold_x = np.log2(threshold + 1)
        xlabel = "log2(raw+1)"

    sns.kdeplot(plot_data, ax=ax, fill=True, alpha=0.5)
    ax.axvline(x=threshold_x, color='red', linestyle='--', linewidth=1.5)

    # Compact threshold label: "T = X" with zero-drop note on the line below
    if log_transform:
        t_label = f"T={threshold:.1f}"
    else:
        t_label = f"T={threshold:.0f} ({threshold_x:.1f})"

    zero_note = f"0s: {n_zeros} ({pct_zeros:.0f}%)" if drop_zeros else ""

    annotation = t_label if not zero_note else f"{t_label}\n{zero_note}"

    # x in data coords, y in axes fraction — reliable regardless of KDE scale
    ax.text(
        threshold_x, 0.97,
        annotation,
        color='red', fontsize=6.5, ha='center', va='top',
        transform=ax.get_xaxis_transform(),
        linespacing=1.3,
    )

    ax.set_title(gene, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.tick_params(labelsize=7)


@saveable_plot()
def fig_mixed_unmixed_cxg_and_corr(
    mixed_results, 
    unmixed_results, 
    inhibitory_genes=None,
    k=None,
    cluster_range=(2, 10),
    corr_vmin=-0.5,
    corr_vmax=0.5,
    cxg_vmin=0,
    cxg_vmax=50,
    figsize=(18, 24),
    log_transform=False,
    dataset=None
):
    """
    Advanced comparison of mixed vs unmixed results with clustering and correlation analysis.
    
    Parameters
    ----------
    mixed_results : pd.DataFrame
        Mixed results with columns: cell_id, gene, spot_count
    unmixed_results : pd.DataFrame
        Unmixed results with columns: cell_id, gene, spot_count
    inhibitory_genes : dict, optional
        Dictionary mapping gene names to threshold counts for defining inhibitory cells.
        A cell is classified as inhibitory if it exceeds the threshold for ANY of these genes (OR logic).
        Default is {'Gad2': 50, 'Sst': 50, 'Npy': 50, 'Pvalb': 50, 'Vip': 50}.
    k : int or None, optional
        Number of clusters to use for all clustering. If None, will search for optimal k
        using silhouette score within cluster_range. Default is None.
    cluster_range : tuple, optional
        Range of k values to test for optimal clustering (min, max). Only used if k is None.
        Default is (2, 10).
    corr_vmin : float, optional
        Minimum value for correlation colormap. Default is -0.5.
    corr_vmax : float, optional
        Maximum value for correlation colormap. Default is 0.5.
    cxg_vmin : float, optional
        Minimum value for cell x gene heatmap colorbar. Default is 0.
    cxg_vmax : float, optional
        Maximum value for cell x gene heatmap colorbar. Default is 50.
    figsize : tuple, optional
        Figure size as (width, height). Default is (18, 16).
    log_transform : bool, optional
        If True, apply log2(x+1) transformation to spot counts. Default is False.
    dataset : HCRDataset, optional
        Required if gene_sort='round_channel'. Used to get channel-gene mapping.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    results : dict
        Dictionary containing:
        - 'mixed_k': optimal k for mixed
        - 'unmixed_k': optimal k for unmixed
        - 'mixed_labels': cluster labels for mixed
        - 'unmixed_labels': cluster labels for unmixed
        - 'mixed_inhibitory_count': number of inhibitory cells in mixed
        - 'unmixed_inhibitory_count': number of inhibitory cells in unmixed
    """
    
    if inhibitory_genes is None:
        inhibitory_genes = {'Gad2': 50, 'Sst': 50, 'Npy': 50, 'Pvalb': 50, 'Vip': 50}

    # Derive colorbar label from log_transform setting
    cbar_label = "Transcript expression log2(x+1)" if log_transform else "Transcript count"

    # --- Data Preparation ---
    # Pivot to cell x gene matrices
    cxg_mixed = mixed_results.pivot(index="cell_id", columns="gene", values="spot_count").fillna(0)
    cxg_unmixed = unmixed_results.pivot(index="cell_id", columns="gene", values="spot_count").fillna(0)
    
    # Apply log transformation if requested
    if log_transform:
        cxg_mixed = np.log2(cxg_mixed + 1)
        cxg_unmixed = np.log2(cxg_unmixed + 1)
    
    # Find optimal k using silhouette score
    def find_optimal_k(data, k_range):
        """Find optimal k using silhouette score."""
        n_samples = len(data)
        if n_samples < 2:
            print(f"Too few samples ({n_samples}) to cluster; using k=1.")
            return 1
        max_k = min(k_range[1], n_samples - 1)
        k_start = min(k_range[0], max_k)
        k_values = range(k_start, max_k + 1)
        silhouette_scores = []
        
        for k_val in k_values:
            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)
        
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = list(k_values)[optimal_idx]
        print(f"All scores: {silhouette_scores}")
        print(f"Optimal k: {optimal_k} (silhouette score: {silhouette_scores[optimal_idx]:.3f})")
        return optimal_k
    
    # Find or use provided k for both datasets
    if k is not None:
        print(f"Using provided k={k} for all clustering")
        mixed_k = k
        unmixed_k = k
    else:
        print("Finding optimal k for mixed data...")
        mixed_k = find_optimal_k(cxg_mixed.values, cluster_range)
        
        print("Finding optimal k for unmixed data...")
        unmixed_k = find_optimal_k(cxg_unmixed.values, cluster_range)
    
    # Perform clustering with optimal k
    kmeans_mixed = KMeans(n_clusters=mixed_k, random_state=42, n_init=10)
    mixed_labels = kmeans_mixed.fit_predict(cxg_mixed.values)
    
    kmeans_unmixed = KMeans(n_clusters=unmixed_k, random_state=42, n_init=10)
    unmixed_labels = kmeans_unmixed.fit_predict(cxg_unmixed.values)

    # Get inhibitory cells for mixed
    print("Identifying mixed inhibitory cells...")
    mixed_inhibitory_mask, mixed_inhib_genes = hcr_filters.get_inhibitory_mask(cxg_mixed, inhibitory_genes)
    
    if mixed_inhibitory_mask.sum() > 0:
        cxg_mixed_inhib = cxg_mixed[mixed_inhibitory_mask]
        mixed_labels_inhib = mixed_labels[mixed_inhibitory_mask]
        print(f"Mixed inhibitory cells: {mixed_inhibitory_mask.sum()}/{len(cxg_mixed)}")
        
        # Find or use provided k for mixed inhibitory cells
        if k is not None:
            mixed_inhib_k = k
        elif len(cxg_mixed_inhib) > cluster_range[1]:
            print("Finding optimal k for mixed inhibitory cells...")
            mixed_inhib_k = find_optimal_k(cxg_mixed_inhib.values, cluster_range)
        else:
            mixed_inhib_k = min(cluster_range[0], len(cxg_mixed_inhib) - 1) if len(cxg_mixed_inhib) > 1 else 1
            print(f"Too few inhibitory cells for clustering, using k={mixed_inhib_k}")
    else:
        print("Warning: No inhibitory cells found in mixed data")
        cxg_mixed_inhib = None
        mixed_labels_inhib = None
        mixed_inhib_k = None
    
    # Get inhibitory cells for unmixed
    print("Identifying unmixed inhibitory cells...")
    unmixed_inhibitory_mask, unmixed_inhib_genes = hcr_filters.get_inhibitory_mask(cxg_unmixed, inhibitory_genes)

    if unmixed_inhibitory_mask.sum() > 0:
        cxg_unmixed_inhib = cxg_unmixed[unmixed_inhibitory_mask]
        unmixed_labels_inhib = unmixed_labels[unmixed_inhibitory_mask]
        print(f"Unmixed inhibitory cells: {unmixed_inhibitory_mask.sum()}/{len(cxg_unmixed)}")
        
        # Find or use provided k for unmixed inhibitory cells
        if k is not None:
            unmixed_inhib_k = k
        elif len(cxg_unmixed_inhib) > cluster_range[1]:
            print("Finding optimal k for unmixed inhibitory cells...")
            unmixed_inhib_k = find_optimal_k(cxg_unmixed_inhib.values, cluster_range)
        else:
            unmixed_inhib_k = min(cluster_range[0], len(cxg_unmixed_inhib) - 1) if len(cxg_unmixed_inhib) > 1 else 1
            print(f"Too few inhibitory cells for clustering, using k={unmixed_inhib_k}")
    else:
        print("Warning: No inhibitory cells found in unmixed data")
        cxg_unmixed_inhib = None
        unmixed_labels_inhib = None
        unmixed_inhib_k = None
    
    # --- Figure Layout ---
    # Rows 0-1: heatmaps, Row 2: correlations, Rows 3-4: inhibitory gene distributions
    # Column count = max(4, n_inh_genes) so distribution rows fit without spanning issues
    inh_genes = list(inhibitory_genes.keys())
    n_inh_genes = len(inh_genes)
    n_cols = max(4, n_inh_genes)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, n_cols, hspace=0.5, wspace=0.4,
                          height_ratios=[3, 3, 2, 1, 1])

    # Rows 0 & 1: use a nested 3-column sub-gridspec so the three heatmaps
    # always span the full figure width, regardless of n_cols.
    gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], wspace=0.3)
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :], wspace=0.3)

    # --- Row 0: Mixed Results ---
    ax00 = fig.add_subplot(gs_row0[0])
    ax01 = fig.add_subplot(gs_row0[1])
    ax02 = fig.add_subplot(gs_row0[2])
    
    # Plot 1: Mixed simple (sorted by Gad2 expression)
    sort_gene = 'Gad2' if 'Gad2' in cxg_mixed.columns else None
    plot_cell_x_gene_simple(cxg_mixed, ax=ax00, title="Mixed Results", gene_sort=sort_gene,
                               clip_range=(cxg_vmin, cxg_vmax), cbar_label=cbar_label)
    ax00.set_yticks([0, len(cxg_mixed)])
    ax00.set_yticklabels([0, len(cxg_mixed)])
    
    # Plot 2: Mixed clustered
    plot_cell_x_gene_clustered(cxg_mixed, ax=ax01, k=mixed_k, 
                                   title=f"Mixed Clustered (k={mixed_k})",
                                   gene_sort='round_channel',
                                   dataset=dataset,
                                   cbar_label=cbar_label,
                                   clip_range=(cxg_vmin, cxg_vmax),
                                   cluster_sort_gene='Gad2',
                                   gene_label="round_channel_gene")
    ax01.set_yticks([0, len(cxg_mixed)])
    ax01.set_yticklabels([0, len(cxg_mixed)])
    
    # Plot 3: Mixed inhibitory clustered
    if cxg_mixed_inhib is not None and len(cxg_mixed_inhib) > 0:
        plot_cell_x_gene_clustered(cxg_mixed_inhib, ax=ax02, k=mixed_inhib_k,
                                    gene_sort='round_channel',
                                       title=f"Mixed Inhibitory Cells (n={len(cxg_mixed_inhib)}, k={mixed_inhib_k})",
                                       dataset=dataset,
                                       cbar_label=cbar_label,
                                       clip_range=(cxg_vmin, cxg_vmax),
                                       cluster_sort_gene='Gad2',
                                       gene_label="round_channel_gene")
        ax02.set_yticks([0, len(cxg_mixed_inhib)])
        ax02.set_yticklabels([0, len(cxg_mixed_inhib)])
    else:
        ax02.text(0.5, 0.5, 'No inhibitory cells', ha='center', va='center', 
                 transform=ax02.transAxes)
        ax02.set_title(f"Mixed Inhibitory Cells")

    # --- Row 1: Unmixed Results ---
    ax10 = fig.add_subplot(gs_row1[0])
    ax11 = fig.add_subplot(gs_row1[1])
    ax12 = fig.add_subplot(gs_row1[2])
    
    # Plot 4: Unmixed simple (sorted by Gad2 expression)
    sort_gene = 'Gad2' if 'Gad2' in cxg_unmixed.columns else None
    plot_cell_x_gene_simple(cxg_unmixed, ax=ax10, title="Unmixed Results", gene_sort=sort_gene,
                               clip_range=(cxg_vmin, cxg_vmax), cbar_label=cbar_label)
    ax10.set_yticks([0, len(cxg_unmixed)])
    ax10.set_yticklabels([0, len(cxg_unmixed)])
    
    # Plot 5: Unmixed clustered
    plot_cell_x_gene_clustered(cxg_unmixed, ax=ax11, k=unmixed_k,
                                   title=f"Unmixed Clustered (k={unmixed_k})",
                                   gene_sort='round_channel',
                                   dataset=dataset,
                                   cbar_label=cbar_label,
                                   clip_range=(cxg_vmin, cxg_vmax),
                                   cluster_sort_gene='Gad2',
                                   gene_label="round_channel_gene")
    ax11.set_yticks([0, len(cxg_unmixed)])
    ax11.set_yticklabels([0, len(cxg_unmixed)])
    
    # Plot 6: Unmixed inhibitory clustered
    if cxg_unmixed_inhib is not None and len(cxg_unmixed_inhib) > 0:
        plot_cell_x_gene_clustered(cxg_unmixed_inhib, ax=ax12, k=unmixed_inhib_k,
                                        gene_sort='round_channel',
                                       title=f"Unmixed Inhibitory Cells (n={len(cxg_unmixed_inhib)}, k={unmixed_inhib_k})",
                                       dataset=dataset,
                                       cbar_label=cbar_label,
                                       clip_range=(cxg_vmin, cxg_vmax),
                                       cluster_sort_gene='Gad2',
                                       gene_label="round_channel_gene")
        ax12.set_yticks([0, len(cxg_unmixed_inhib)])
        ax12.set_yticklabels([0, len(cxg_unmixed_inhib)])
    else:
        ax12.text(0.5, 0.5, 'No inhibitory cells', ha='center', va='center',
                 transform=ax12.transAxes)
        ax12.set_title(f"Unmixed Inhibitory Cells")
    
    # --- Row 2: Correlation Matrices (4 plots) ---
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2, :], wspace=0.4)
    ax20 = fig.add_subplot(gs_row2[0])
    ax21 = fig.add_subplot(gs_row2[1])
    ax22 = fig.add_subplot(gs_row2[2])
    ax23 = fig.add_subplot(gs_row2[3])
    
    # Compute gene-gene pairwise correlations (correlate columns, which are genes)
    corr_unmixed = cxg_unmixed.corr(method='pearson')
    corr_mixed = cxg_mixed.corr(method='pearson')
    
    # Plot 1: Unmixed all cells correlation
    im_unmixed = ax20.imshow(corr_unmixed.values, cmap='RdBu_r', 
                            vmin=corr_vmin, vmax=corr_vmax,
                            aspect='auto', interpolation='nearest')
    ax20.set_aspect('equal', adjustable='box')
    ax20.set_title("Unmixed All Cells\nGene-Gene Correlation", fontsize=10)
    ax20.set_xlabel("Gene", fontsize=8)
    ax20.set_ylabel("Gene", fontsize=8)
    ax20.set_xticks(range(len(corr_unmixed.columns)))
    ax20.set_yticks(range(len(corr_unmixed.columns)))
    ax20.set_xticklabels(corr_unmixed.columns, rotation=90, fontsize=7)
    ax20.set_yticklabels(corr_unmixed.columns, fontsize=7)
    plt.colorbar(im_unmixed, ax=ax20, fraction=0.046, pad=0.04)
    
    # Plot 2: Mixed all cells correlation
    im_mixed = ax21.imshow(corr_mixed.values, cmap='RdBu_r',
                          vmin=corr_vmin, vmax=corr_vmax,
                          aspect='auto', interpolation='nearest')
    ax21.set_aspect('equal', adjustable='box')
    ax21.set_title("Mixed All Cells\nGene-Gene Correlation", fontsize=10)
    ax21.set_xlabel("Gene", fontsize=8)
    ax21.set_ylabel("Gene", fontsize=8)
    ax21.set_xticks(range(len(corr_mixed.columns)))
    ax21.set_yticks(range(len(corr_mixed.columns)))
    ax21.set_xticklabels(corr_mixed.columns, rotation=90, fontsize=7)
    ax21.set_yticklabels(corr_mixed.columns, fontsize=7)
    plt.colorbar(im_mixed, ax=ax21, fraction=0.046, pad=0.04)
    
    # Plot 3: Unmixed inhibitory cells correlation
    if cxg_unmixed_inhib is not None and len(cxg_unmixed_inhib) > 1:
        corr_unmixed_inhib = cxg_unmixed_inhib.corr(method='pearson')
        im_unmixed_inhib = ax22.imshow(corr_unmixed_inhib.values, cmap='RdBu_r',
                                      vmin=corr_vmin, vmax=corr_vmax,
                                      aspect='auto', interpolation='nearest')
        ax22.set_aspect('equal', adjustable='box')
        ax22.set_title(f"Unmixed Inhibitory\n(n={len(cxg_unmixed_inhib)}) Correlation", fontsize=10)
        ax22.set_xlabel("Gene", fontsize=8)
        ax22.set_ylabel("Gene", fontsize=8)
        ax22.set_xticks(range(len(corr_unmixed_inhib.columns)))
        ax22.set_yticks(range(len(corr_unmixed_inhib.columns)))
        ax22.set_xticklabels(corr_unmixed_inhib.columns, rotation=90, fontsize=7)
        ax22.set_yticklabels(corr_unmixed_inhib.columns, fontsize=7)
        plt.colorbar(im_unmixed_inhib, ax=ax22, fraction=0.046, pad=0.04)
    else:
        ax22.text(0.5, 0.5, 'No/insufficient\ninhibitory cells', ha='center', va='center',
                 transform=ax22.transAxes, fontsize=10)
        ax22.set_title(f"Unmixed Inhibitory\nCorrelation", fontsize=10)
        ax22.axis('off')
    
    # Plot 4: Mixed inhibitory cells correlation
    if cxg_mixed_inhib is not None and len(cxg_mixed_inhib) > 1:
        corr_mixed_inhib = cxg_mixed_inhib.corr(method='pearson')
        im_mixed_inhib = ax23.imshow(corr_mixed_inhib.values, cmap='RdBu_r',
                                    vmin=corr_vmin, vmax=corr_vmax,
                                    aspect='auto', interpolation='nearest')
        ax23.set_aspect('equal', adjustable='box')
        ax23.set_title(f"Mixed Inhibitory\n(n={len(cxg_mixed_inhib)}) Correlation", fontsize=10)
        ax23.set_xlabel("Gene", fontsize=8)
        ax23.set_ylabel("Gene", fontsize=8)
        ax23.set_xticks(range(len(corr_mixed_inhib.columns)))
        ax23.set_yticks(range(len(corr_mixed_inhib.columns)))
        ax23.set_xticklabels(corr_mixed_inhib.columns, rotation=90, fontsize=7)
        ax23.set_yticklabels(corr_mixed_inhib.columns, fontsize=7)
        plt.colorbar(im_mixed_inhib, ax=ax23, fraction=0.046, pad=0.04)
    else:
        ax23.text(0.5, 0.5, 'No/insufficient\ninhibitory cells', ha='center', va='center',
                 transform=ax23.transAxes, fontsize=10)
        ax23.set_title(f"Mixed Inhibitory\nCorrelation", fontsize=10)
        ax23.axis('off')
    
    # --- Row 3: Mixed inhibitory gene distributions ---
    for col_idx, gene in enumerate(inh_genes):
        ax = fig.add_subplot(gs[3, col_idx])
        threshold = inhibitory_genes[gene]
        _plot_inhibitory_gene_dist(cxg_mixed, gene, threshold, log_transform, ax=ax)
        if col_idx == 0:
            ax.set_ylabel("Density\n(Mixed)", fontsize=8)

    # --- Row 4: Unmixed inhibitory gene distributions ---
    for col_idx, gene in enumerate(inh_genes):
        ax = fig.add_subplot(gs[4, col_idx])
        threshold = inhibitory_genes[gene]
        _plot_inhibitory_gene_dist(cxg_unmixed, gene, threshold, log_transform,ax=ax)
        if col_idx == 0:
            ax.set_ylabel("Density\n(Unmixed)", fontsize=8)

    # Overall title
    fig.suptitle('Mixed vs Unmixed Comparison with Clustering and Correlation Analysis', 
                fontsize=14, y=0.995)
    
    # Prepare results dictionary
    results = {
        'mixed_k': mixed_k,
        'unmixed_k': unmixed_k,
        'mixed_labels': mixed_labels,
        'unmixed_labels': unmixed_labels,
        'mixed_inhibitory_count': mixed_inhibitory_mask.sum() if cxg_mixed_inhib is not None else 0,
        'unmixed_inhibitory_count': unmixed_inhibitory_mask.sum() if cxg_unmixed_inhib is not None else 0
    }
    
    return fig, results


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
    # fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=fig_size, gridspec_kw={"width_ratios": [3, 1]})

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

    # ax_scatter.set_aspect("equal", adjustable="box")
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
    cluster_df = pd.DataFrame(
        {"Cluster": np.arange(len(cluster_sizes)), "Size": cluster_sizes, "Percentage": cluster_percentages}
    )
    return cluster_df


def plot_cluster_centroids(cell_info_clusters, cluster_n, save=False, output_dir=Path("/root/capsule/scratch/")):
    """
    Plot the centroids of a specific cluster.

    Parameters:
    cluster_n (int): The cluster number to plot.
    """
    # Filter the cell_info_clusters DataFrame for the specified cluster
    plot_cluster_df = cell_info_clusters[cell_info_clusters["cluster_label"] == cluster_n]

    # Set the cluster label to 1 for plotting purposes
    plot_cluster_df["cluster_label"] = 1

    # Get the maximum x and y coordinates for setting limits
    max_x = cell_info_clusters["x_centroid"].max()
    max_y = cell_info_clusters["y_centroid"].max()

    # Plot the centroids with histogram
    fig = plot_centroids_with_hist(
        plot_cluster_df,
        orientation="XY",
        color_col="cluster_label",
        cmap="Greys_r",
        xlims=(0, max_x),
        ylims=(0, max_y),
        show_colorbar=False,
        fig_size=(4, 4),
        save=save,
        output_dir=output_dir,
        title_str=f"Cluster {cluster_n}",
    )

    # if not save:
    #     #plt.show()

    return fig



