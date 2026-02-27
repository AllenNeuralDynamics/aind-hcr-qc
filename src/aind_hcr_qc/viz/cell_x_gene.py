import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list, fcluster
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
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


def cluster_cells(
    cxg,
    method="kmeans",
    k=None,
    cluster_sort_gene="Gad2",
    random_state=42,
    linkage_method="ward",
    metric="euclidean",
    n_clusters_ward=None,
):
    """
    Cluster (or order) cells in a pre-processed cell × gene matrix.

    Parameters
    ----------
    cxg : pd.DataFrame
        Pre-processed cell × gene matrix.  Should already have NaNs filled,
        be cast to int and clipped — i.e. the same state as the data inside
        ``plot_cell_x_gene_clustered`` just before the clustering block.
    method : {'kmeans', 'agglomerative', 'ward_leaf'}
        Clustering algorithm to use:

        ``'kmeans'``
            Scikit-learn KMeans.  Requires ``k``.
        ``'agglomerative'``
            Scikit-learn AgglomerativeClustering with Ward linkage.  Requires ``k``.
            Clusters are subsequently sorted by mean expression of
            ``cluster_sort_gene`` (ascending), matching KMeans behaviour.
        ``'ward_leaf'``
            SciPy hierarchical Ward linkage with optimal-leaf ordering.
            Produces a continuous row ordering that groups similar cells
            together — no discrete cluster IDs are assigned unless you also
            pass ``n_clusters_ward``.  ``cluster_sort_gene`` is ignored.
    k : int, optional
        Number of clusters.  Required for ``'kmeans'`` and ``'agglomerative'``.
        Ignored by ``'ward_leaf'``.
    cluster_sort_gene : str, optional
        After ``'kmeans'`` or ``'agglomerative'`` clustering, reorder the
        cluster IDs so that cluster 0 has the *lowest* mean expression of
        this gene and the highest-numbered cluster has the most.  Falls back
        to total mean expression when the gene is absent.  Default ``'Gad2'``.
    random_state : int, optional
        Random seed for KMeans (default 42).
    linkage_method : str, optional
        Linkage criterion passed to
        ``scipy.cluster.hierarchy.linkage`` (default ``'ward'``).
        Only used by ``'ward_leaf'``.
    metric : str, optional
        Distance metric passed to
        ``scipy.cluster.hierarchy.linkage`` (default ``'euclidean'``).
        Only used by ``'ward_leaf'``.
    n_clusters_ward : int, optional
        If provided, cut the ``'ward_leaf'`` dendrogram into this many flat
        clusters via ``scipy.cluster.hierarchy.fcluster``.  The resulting
        integer cluster IDs are returned in ``cluster_labels``; if ``None``
        (default) ``cluster_labels`` will be ``None``.

    Returns
    -------
    sorted_cxg : pd.DataFrame
        A copy of ``cxg`` with rows reordered according to the clustering.
    cluster_labels : np.ndarray or None
        Integer cluster assignment for each row of ``sorted_cxg``
        (same length, same order).  ``None`` for ``'ward_leaf'`` when
        ``n_clusters_ward`` is not supplied.
    sorted_cell_ids : pd.Index
        Row index of ``sorted_cxg`` — useful for downstream alignment.
    """
    if method in ("kmeans", "agglomerative") and k is None:
        raise ValueError(f"method='{method}' requires k to be specified.")

    cxg = cxg.copy()

    if method == "kmeans":
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        raw_labels = kmeans.fit_predict(cxg.values)

    elif method == "agglomerative":
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        raw_labels = agg.fit_predict(cxg.values)

    elif method == "ward_leaf":
        Z = linkage(cxg.values, method=linkage_method, metric=metric)
        Z_opt = optimal_leaf_ordering(Z, cxg.values)
        ordered_indices = leaves_list(Z_opt)
        sorted_cxg = cxg.iloc[ordered_indices].copy()

        if n_clusters_ward is not None:
            flat = fcluster(Z_opt, t=n_clusters_ward, criterion="maxclust")
            # flat labels are 1-based and in original row order; reindex to sorted order
            cluster_labels = flat[ordered_indices] - 1  # make 0-based
        else:
            cluster_labels = None

        return sorted_cxg, cluster_labels, sorted_cxg.index

    else:
        raise ValueError(
            f"Unknown clustering method '{method}'. "
            "Choose from: 'kmeans', 'agglomerative', 'ward_leaf'."
        )

    # --- Shared post-processing for kmeans / agglomerative ---
    # Sort clusters by mean expression of cluster_sort_gene (ascending)
    cxg["_cluster"] = raw_labels
    if cluster_sort_gene and cluster_sort_gene in cxg.columns:
        cluster_means = cxg.groupby("_cluster")[cluster_sort_gene].mean()
    else:
        cluster_means = (
            cxg.drop(columns=["_cluster"])
            .groupby(raw_labels)
            .mean()
            .mean(axis=1)
        )

    sorted_cluster_ids = cluster_means.sort_values(ascending=True).index
    rank_map = {orig: rank for rank, orig in enumerate(sorted_cluster_ids)}
    cxg["_cluster_rank"] = cxg["_cluster"].map(rank_map)
    cxg = cxg.sort_values("_cluster_rank", ascending=True)

    cluster_labels = cxg["_cluster_rank"].values
    sorted_cell_ids = cxg.index
    cxg = cxg.drop(columns=["_cluster", "_cluster_rank"])

    return cxg, cluster_labels, sorted_cell_ids


@saveable_plot()
def plot_cell_x_gene_clustered(
    cxg,
    clip_range=(0, 50),
    sort_gene=None,
    fig_size=(4, 6),
    k=3,
    cluster_method="kmeans",
    cluster_result=None,
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
    Plot the cell x gene matrix as an image with inverted colormap and clustering.

    Clustering is delegated to :func:`cluster_cells`, which supports three methods:
    ``'kmeans'``, ``'agglomerative'``, and ``'ward_leaf'``.  You can also pass a
    pre-computed ``cluster_result`` tuple to skip clustering entirely.

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
        Number of clusters. Required for ``'kmeans'`` and ``'agglomerative'``.
        Ignored for ``'ward_leaf'``. Default is 3.
    cluster_method : {'kmeans', 'agglomerative', 'ward_leaf'}, optional
        Clustering algorithm to use (default ``'kmeans'``). See :func:`cluster_cells`
        for full details of each option.
    cluster_result : tuple, optional
        Pre-computed result from :func:`cluster_cells` as
        ``(sorted_cxg, cluster_labels, sorted_cell_ids)``.  When supplied the
        function skips clustering and uses this directly.  ``cxg`` must still
        be passed for gene-column sorting.
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
        After ``'kmeans'`` or ``'agglomerative'`` clustering, order the clusters by their
        mean expression of this gene (ascending). Default is ``'Gad2'``. Falls back to
        total mean expression if the gene is absent. Ignored for ``'ward_leaf'``.
    gene_label : {'gene', 'round_channel_gene'}, optional
        Column from the channel-gene table to use as x-axis tick labels.
        'gene' (default) shows only the gene name; 'round_channel_gene' shows
        e.g. "R2_ch647_Gad2". Requires ``dataset`` when not 'gene'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    cluster_labels : np.ndarray or None
        Array of cluster assignments for each cell (None for ``'ward_leaf'`` without
        ``n_clusters_ward``, or when cells are sorted by a gene).
    sorted_cell_ids : pd.Index
        Index of cell IDs in the same sorted order as the plot rows.
    """
    if not isinstance(cxg, pd.DataFrame):
        raise ValueError("Input cxg must be a pandas DataFrame.")

    cxg = cxg.copy()  # avoid modifying the original DataFrame
    cxg = cxg.fillna(0)
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
        pass  # gene name — used for row sorting below
    elif gene_sort is not None and gene_sort != 'alphabetical':
        raise ValueError(f"gene_sort '{gene_sort}' not recognized. Use 'alphabetical', 'round_channel', or a gene name.")

    # Build x-axis tick labels (after column order is finalised)
    x_labels = _build_gene_labels(cxg.columns.tolist(), gene_label, dataset)

    # Determine row ordering
    cell_sort_gene = gene_sort if gene_sort in cxg.columns else sort_gene

    if cell_sort_gene is not None and cell_sort_gene in cxg.columns:
        # Simple gene-expression sort — no clustering
        cxg = cxg.sort_values(by=cell_sort_gene, ascending=False)
        cluster_labels = None
        sorted_cell_ids = cxg.index
        y_label = f"Cells sorted by {cell_sort_gene}"
    elif cluster_result is not None:
        # Use pre-computed clustering result
        sorted_cxg, cluster_labels, sorted_cell_ids = cluster_result
        # Reindex columns to match the gene sort applied above
        cxg = sorted_cxg.reindex(columns=cxg.columns, fill_value=0)
        sort_label = cluster_sort_gene if cluster_sort_gene else "pre-computed"
        y_label = f"Cells ({cluster_method} pre-computed)"
    else:
        # Delegate to cluster_cells()
        cxg, cluster_labels, sorted_cell_ids = cluster_cells(
            cxg,
            method=cluster_method,
            k=k,
            cluster_sort_gene=cluster_sort_gene,
        )
        if cluster_method == "ward_leaf":
            y_label = "Cells (Ward optimal-leaf order)"
        else:
            sort_label = cluster_sort_gene if cluster_sort_gene else "total expr"
            y_label = f"Cells ({cluster_method}, sorted by mean {sort_label} ↑)"

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

    # Add cluster boundary lines and labels if clustering was performed and requested
    if cluster_labels is not None and add_cluster_labels:
        cluster_changes = np.where(np.diff(cluster_labels) != 0)[0] + 0.5

        for boundary in cluster_changes:
            ax.axhline(y=boundary, color="green", linestyle="--", linewidth=2.0, alpha=0.9)

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
        - 'mixed_labels': raw KMeans cluster labels for mixed (aligned to cxg_mixed_pivot row order)
        - 'unmixed_labels': raw KMeans cluster labels for unmixed (aligned to cxg_unmixed_pivot row order)
        - 'mixed_labels_ranked': cluster labels remapped by mean Gad2 expression rank (same row order)
        - 'unmixed_labels_ranked': cluster labels remapped by mean Gad2 expression rank (same row order)
        - 'mixed_sorted_cell_ids': cell_id index in plot row order (sorted by cluster rank)
        - 'unmixed_sorted_cell_ids': cell_id index in plot row order (sorted by cluster rank)
        - 'cxg_mixed_pivot': pivot cell x gene DataFrame for mixed (cell_id x gene)
        - 'cxg_unmixed_pivot': pivot cell x gene DataFrame for unmixed (cell_id x gene)
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
    _, mixed_labels_ranked, mixed_sorted_cell_ids = plot_cell_x_gene_clustered(
                                   cxg_mixed, ax=ax01, k=mixed_k, 
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
    _, unmixed_labels_ranked, unmixed_sorted_cell_ids = plot_cell_x_gene_clustered(
                                   cxg_unmixed, ax=ax11, k=unmixed_k,
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
    
    # Reindex pivots to match the row order that ranked labels were computed against.
    # plot_cell_x_gene_clustered sorts rows internally; mixed_labels_ranked[i] corresponds
    # to mixed_sorted_cell_ids[i], NOT to the original cxg_mixed row order.
    cxg_mixed_pivot_sorted   = cxg_mixed.loc[mixed_sorted_cell_ids]
    cxg_unmixed_pivot_sorted = cxg_unmixed.loc[unmixed_sorted_cell_ids]

    # Prepare results dictionary
    results = {
        'mixed_k': mixed_k,
        'unmixed_k': unmixed_k,
        'mixed_labels': mixed_labels,
        'unmixed_labels': unmixed_labels,
        'mixed_labels_ranked': mixed_labels_ranked,
        'unmixed_labels_ranked': unmixed_labels_ranked,
        'mixed_sorted_cell_ids': mixed_sorted_cell_ids,
        'unmixed_sorted_cell_ids': unmixed_sorted_cell_ids,
        'cxg_mixed_pivot': cxg_mixed_pivot_sorted,
        'cxg_unmixed_pivot': cxg_unmixed_pivot_sorted,
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


# =================================================================================================
# Thresholding Section
# =================================================================================================
#
# Functions for fitting Gaussian Mixture Models (GMM) to cell x gene spot count distributions
# in order to identify expression thresholds that separate signal from noise.
#
# Workflow:
#   1. compute_gmm_bic_table          — compare 1- vs 2-component fits per gene
#   2. plot_bic_comparison            — visualise delta-BIC as a bar chart
#   3. fit_gmm_threshold              — fit a 2-component GMM to one gene and find the crossover
#   4. threshold_genes                — batch-apply fit_gmm_threshold across a selection of genes
#   5. plot_gmm_fit                   — single-gene QC plot (histogram + GMM curves + threshold)
#   6. plot_gmm_threshold_grid        — grid of QC plots for all selected genes
#   7. fit_gmm_marker_via_subgenes      — refit a marker gene anchored by confident sub-gene cells
#   8. plot_gmm_marker_subgene_analysis — two-panel QC figure for the sub-gene-anchored refit
#   9. run_inhibitory_gmm_thresholding  — end-to-end pipeline: filter → fit → refit → combined thresholds
#  10. filter_cells_by_gmm_thresholds   — keep only cells above threshold for any/all genes
#
# All functions operate on log2(raw_counts + 1) transformed values internally.
# Returned thresholds are provided in both log2 and raw (back-transformed) space.
# =================================================================================================


def _log2_transform(values: np.ndarray) -> np.ndarray:
    """Apply log2(x + 1) transform to a 1-D array of raw counts."""
    return np.log2(values + 1.0)


def _back_transform(log2_values: np.ndarray) -> np.ndarray:
    """Invert log2(x + 1) back to raw count space: 2^x - 1."""
    return np.power(2.0, log2_values) - 1.0


def _prepare_gene_values(pivot_df: pd.DataFrame, gene: str, remove_zeros: bool) -> np.ndarray:
    """
    Extract, optionally filter, and log2-transform values for one gene.

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Cell × gene pivot table with raw spot counts (cells as rows, genes as columns).
    gene : str
        Column name to extract.
    remove_zeros : bool
        If True, remove zero-count cells before transforming.

    Returns
    -------
    np.ndarray
        Log2(x + 1) transformed 1-D array ready for GMM fitting.
    """
    raw = pivot_df[gene].values.astype(float)
    if remove_zeros:
        n_zeros = int((raw == 0).sum())
        if n_zeros > 0:
            warnings.warn(
                f"[GMM] '{gene}': removing {n_zeros} / {len(raw)} zero-count cells before fitting.",
                stacklevel=3,
            )
        raw = raw[raw > 0]
    return _log2_transform(raw)


# -------------------------------------------------------------------------------------------------
# 1. BIC comparison
# -------------------------------------------------------------------------------------------------


def compute_gmm_bic_table(
    pivot_df: pd.DataFrame,
    genes: list = None,
    remove_zeros: bool = True,
) -> pd.DataFrame:
    """
    Fit 1- and 2-component GMMs to each gene's spot-count distribution and return BIC scores.

    A positive ``delta_bic`` (= bic_1 − bic_2) indicates the 2-component model explains
    the data better; use this to decide whether GMM thresholding is warranted for a gene.

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Cell × gene pivot table with raw spot counts.
    genes : list of str, optional
        Subset of genes to evaluate. Defaults to all columns in ``pivot_df``.
    remove_zeros : bool, optional
        Remove zero-count cells before fitting (default True).

    Returns
    -------
    pd.DataFrame
        Columns: ``gene``, ``bic_1``, ``bic_2``, ``delta_bic``.
    """
    genes = genes if genes is not None else pivot_df.columns.tolist()
    rows = []
    for gene in genes:
        log_vals = _prepare_gene_values(pivot_df, gene, remove_zeros)
        X = log_vals.reshape(-1, 1)
        gmm1 = GaussianMixture(n_components=1, covariance_type="full", n_init=5, random_state=42)
        gmm2 = GaussianMixture(n_components=2, covariance_type="full", n_init=10, random_state=42)
        gmm1.fit(X)
        gmm2.fit(X)
        bic1, bic2 = gmm1.bic(X), gmm2.bic(X)
        rows.append({"gene": gene, "bic_1": bic1, "bic_2": bic2, "delta_bic": bic1 - bic2})

    bic_df = pd.DataFrame(rows)
    n_better = (bic_df["delta_bic"] > 0).sum()
    print(f"[GMM] BIC: 2-component model preferred for {n_better} / {len(bic_df)} gene(s).")
    return bic_df


# -------------------------------------------------------------------------------------------------
# 2. BIC bar chart
# -------------------------------------------------------------------------------------------------


def plot_bic_comparison(
    bic_df: pd.DataFrame,
    ax=None,
) -> tuple:
    """
    Bar chart of delta-BIC (bic_1 − bic_2) per gene.

    Positive bars indicate the 2-component GMM is preferred over the 1-component model.
    A horizontal dashed line at zero is drawn for reference.

    Parameters
    ----------
    bic_df : pd.DataFrame
        Output of :func:`compute_gmm_bic_table`.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on.  A new figure is created when ``ax`` is None.

    Returns
    -------
    fig, ax
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(max(4, len(bic_df) * 0.7), 4))
    else:
        fig = ax.get_figure()

    colors = ["#4C7A9A" if v > 0 else "#C0504D" for v in bic_df["delta_bic"]]
    ax.bar(bic_df["gene"], bic_df["delta_bic"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Gene")
    ax.set_ylabel("ΔBIC  (BIC₁ − BIC₂)")
    ax.set_title("GMM model comparison: ΔBIC per gene\n(positive → 2-component preferred)")
    plt.xticks(rotation=45, ha="right")
    if created_fig:
        fig.tight_layout()
    return fig, ax


# -------------------------------------------------------------------------------------------------
# 3. Single-gene threshold fitting
# -------------------------------------------------------------------------------------------------


def fit_gmm_threshold(
    values: np.ndarray,
    gene: str = "",
    n_components: int = 2,
    threshold_prob: float = 0.5,
    remove_zeros: bool = True,
    covariance_type: str = "full",
) -> dict:
    """
    Fit a GMM to a 1-D array of raw spot counts for one gene and find the expression threshold.

    Internally applies log2(x + 1) and optionally removes zeros before fitting.

    Parameters
    ----------
    values : np.ndarray
        Raw spot counts for one gene across all cells.
    gene : str, optional
        Gene name used only for warning messages.
    n_components : int, optional
        Number of GMM components (default 2).
    threshold_prob : float, optional
        Posterior probability of the signal component at the threshold (default 0.5).
    remove_zeros : bool, optional
        Remove zero-count cells before fitting (default True).
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, optional
        GMM covariance structure passed to sklearn (default 'full').
        Use 'tied' to constrain both components to share the same covariance,
        which can stabilise fits on smaller or noisier populations.

    Returns
    -------
    dict with keys:
        ``threshold_log2``    – threshold in log2(count + 1) space
        ``threshold_raw``     – threshold back-transformed to raw count space
        ``gmm``               – fitted :class:`~sklearn.mixture.GaussianMixture` object
        ``log_values``        – log2-transformed values used for fitting
        ``covariance_type``   – covariance type used for the fit
    """
    raw = values.astype(float)
    if remove_zeros:
        n_zeros = int((raw == 0).sum())
        if n_zeros > 0:
            warnings.warn(
                f"[GMM] '{gene}': removing {n_zeros} / {len(raw)} zero-count cells before fitting.",
                stacklevel=2,
            )
        raw = raw[raw > 0]

    log_vals = _log2_transform(raw)
    X = log_vals.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        means_init=[[3.0], [8.0]],
        n_init=10,
        random_state=42,
    )
    gmm.fit(X)

    signal_component = int(np.argmax(gmm.means_.flatten()))

    x_range = np.linspace(log_vals.min(), log_vals.max(), 10_000).reshape(-1, 1)
    probs = gmm.predict_proba(x_range)
    signal_probs = probs[:, signal_component]
    crossover_idx = np.where(signal_probs >= threshold_prob)[0]

    if len(crossover_idx) == 0:
        threshold_log2 = float(log_vals.max())
    else:
        threshold_log2 = float(x_range[crossover_idx[0], 0])

    threshold_raw = float(_back_transform(np.array([threshold_log2]))[0])
    return {
        "threshold_log2": threshold_log2,
        "threshold_raw": threshold_raw,
        "gmm": gmm,
        "log_values": log_vals,
        "covariance_type": covariance_type,
    }


# -------------------------------------------------------------------------------------------------
# 4. Batch thresholding
# -------------------------------------------------------------------------------------------------


def threshold_genes(
    pivot_df: pd.DataFrame,
    genes: list = None,
    remove_zeros: bool = True,
    threshold_prob: float = 0.5,
    covariance_type: str = "full",
) -> pd.DataFrame:
    """
    Fit GMM thresholds for a selection of genes in a cell × gene pivot table.

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Cell × gene pivot table with raw spot counts.
    genes : list of str, optional
        Genes to threshold. Defaults to all columns.
    remove_zeros : bool, optional
        Remove zero-count cells before fitting (default True).
    threshold_prob : float, optional
        Posterior probability at the threshold crossover (default 0.5).
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, optional
        GMM covariance structure (default 'full').

    Returns
    -------
    pd.DataFrame
        Columns: ``gene``, ``threshold_log2``, ``threshold_raw``.
    """
    genes = genes if genes is not None else pivot_df.columns.tolist()

    # Validate all requested genes are present before fitting anything
    missing = [g for g in genes if g not in pivot_df.columns]
    if missing:
        available = sorted(pivot_df.columns.tolist())
        raise KeyError(
            f"[GMM] The following inhibitory marker gene(s) were requested but are "
            f"not present in the cell × gene pivot table: {missing}.\n"
            f"Available genes: {available}\n"
            f"Check that the correct rounds/channels were loaded and that gene name "
            f"spelling matches exactly (case-sensitive)."
        )

    rows = []
    for gene in genes:
        result = fit_gmm_threshold(
            pivot_df[gene].values,
            gene=gene,
            remove_zeros=remove_zeros,
            threshold_prob=threshold_prob,
            covariance_type=covariance_type,
        )
        rows.append(
            {
                "gene": gene,
                "threshold_log2": result["threshold_log2"],
                "threshold_raw": result["threshold_raw"],
            }
        )

    thresholds_df = pd.DataFrame(rows)
    print(f"[GMM] Thresholds computed for {len(thresholds_df)} gene(s).")
    return thresholds_df


# -------------------------------------------------------------------------------------------------
# 5. Single-gene QC plot
# -------------------------------------------------------------------------------------------------


def plot_gmm_fit(
    ax,
    log_values: np.ndarray,
    gene: str,
    gmm,
    threshold_log2: float,
    annotate_threshold: bool = True,
) -> None:
    """
    Plot the GMM fit for one gene on an existing axis.

    Shows a histogram of log2(count + 1) values, individual GMM component curves,
    the combined mixture density, and a vertical threshold line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    log_values : np.ndarray
        Log2-transformed values used for fitting.
    gene : str
        Gene name for the subplot title.
    gmm : GaussianMixture
        Fitted GMM object.
    threshold_log2 : float
        Threshold in log2 space.
    annotate_threshold : bool, optional
        If True, annotate the threshold value (log2 and raw) on the plot (default True).
    """
    from scipy.stats import norm

    x_plot = np.linspace(log_values.min(), log_values.max(), 500).reshape(-1, 1)

    # Histogram (normalised to density)
    ax.hist(log_values, bins=40, density=True, color="#AABDCC", edgecolor="white",
            linewidth=0.4, alpha=0.8, label="data")

    # Per-component curves
    component_colors = ["#E07B54", "#5B8DB8"]
    for k in range(gmm.n_components):
        weight = gmm.weights_[k]
        mean = gmm.means_[k, 0]
        # 'tied' → single shared covariance, shape (1, 1); all others per-component (n, 1, 1)
        if gmm.covariance_type == "tied":
            std = np.sqrt(gmm.covariances_[0, 0])
        else:
            std = np.sqrt(gmm.covariances_[k, 0, 0])
        component_density = weight * norm.pdf(x_plot.flatten(), mean, std)
        ax.plot(x_plot.flatten(), component_density, color=component_colors[k % 2],
                linewidth=1.5, linestyle="--", alpha=0.85)

    # Full mixture density
    log_prob = gmm.score_samples(x_plot)
    mixture_density = np.exp(log_prob)
    ax.plot(x_plot.flatten(), mixture_density, color="black", linewidth=1.5, label="mixture")

    # Threshold line
    ax.axvline(threshold_log2, color="#C0504D", linewidth=1.5, linestyle="-", label="threshold")

    if annotate_threshold:
        threshold_raw = float(_back_transform(np.array([threshold_log2]))[0])
        ax.text(
            threshold_log2 + 0.05,
            ax.get_ylim()[1] * 0.85,
            f"log₂: {threshold_log2:.2f}\nraw: {threshold_raw:.1f}",
            fontsize=7,
            color="#C0504D",
            va="top",
        )

    ax.set_title(gene, fontsize=9)
    ax.set_xlabel("log₂(count + 1)", fontsize=7)
    ax.set_ylabel("density", fontsize=7)
    ax.tick_params(labelsize=6)


# -------------------------------------------------------------------------------------------------
# 6. QC grid plot
# -------------------------------------------------------------------------------------------------

@saveable_plot()
def plot_gmm_threshold_grid(
    pivot_df: pd.DataFrame,
    genes: list = None,
    remove_zeros: bool = True,
    ncols: int = 4,
    annotate_threshold: bool = True,
    threshold_prob: float = 0.5,
    covariance_type: str = "full",
) -> plt.Figure:
    """
    Grid of GMM fit QC plots for a selection of genes.

    For each gene, fits a 2-component GMM via :func:`fit_gmm_threshold` and renders a
    histogram with fitted component curves and the crossover threshold.

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Cell × gene pivot table with raw spot counts.
    genes : list of str, optional
        Genes to plot. Defaults to all columns.
    remove_zeros : bool, optional
        Remove zero-count cells before fitting (default True).
    ncols : int, optional
        Number of columns in the grid (default 4).
    annotate_threshold : bool, optional
        Annotate each subplot with the threshold value (default True).
    threshold_prob : float, optional
        Posterior probability at the threshold crossover (default 0.5).
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, optional
        GMM covariance structure (default 'full').

    Returns
    -------
    matplotlib.figure.Figure
    """
    genes = genes if genes is not None else pivot_df.columns.tolist()
    n = len(genes)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.5, nrows * 3.0),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()

    print(f"[GMM] Fitting and plotting {n} gene(s)...")
    for i, gene in enumerate(genes):
        result = fit_gmm_threshold(
            pivot_df[gene].values,
            gene=gene,
            remove_zeros=remove_zeros,
            threshold_prob=threshold_prob,
            covariance_type=covariance_type,
        )
        plot_gmm_fit(
            axes_flat[i],
            result["log_values"],
            gene,
            result["gmm"],
            result["threshold_log2"],
            annotate_threshold=annotate_threshold,
        )

    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("GMM threshold QC", fontsize=11, y=1.01)
    return fig


# -------------------------------------------------------------------------------------------------
# 7. Sub-gene-anchored GMM refit  (e.g. Gad2 anchored by Pvalb / Sst / Vip)
# -------------------------------------------------------------------------------------------------


def fit_gmm_marker_via_subgenes(
    pivot_df: pd.DataFrame,
    marker_gene: str,
    subgenes: list,
    remove_zeros: bool = True,
    threshold_prob: float = 0.5,
    mask_logic: str = "any",
    covariance_type: str = "full",
) -> dict:
    """
    Refit a GMM threshold for a broad marker gene anchored by confident sub-gene positive cells.

    **Rationale:** A broad marker (e.g. Gad2) is expressed across a heterogeneous population
    whose non-expressing cells can dominate and obscure the signal component. Sub-type specific
    genes (e.g. Pvalb, Sst, Vip) identify cells that are very likely true positives. By
    restricting the marker refit to cells that are positive for at least one (or all) sub-genes,
    the signal component is anchored to biologically confirmed expressors, yielding a cleaner
    threshold.

    Steps
    -----
    1. Threshold each sub-gene independently via :func:`fit_gmm_threshold`.
    2. Build a boolean positive mask per sub-gene (raw count > threshold_raw).
    3. Combine masks using ``mask_logic`` ('any' = OR, 'all' = AND).
    4. Extract ``marker_gene`` raw counts for masked cells only.
    5. Refit GMM on the restricted population and return the threshold.

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Cell × gene pivot table with raw spot counts (cell_id as index).
    marker_gene : str
        The broad marker gene to refit (e.g. 'Gad2', 'GFP').
    subgenes : list of str
        Sub-type genes used to build the positive-cell mask (e.g. ['Pvalb', 'Sst', 'Vip']).
    remove_zeros : bool, optional
        Remove zero-count cells before fitting sub-gene and marker GMMs (default True).
    threshold_prob : float, optional
        Posterior probability at the threshold crossover (default 0.5).
    mask_logic : {'any', 'all'}, optional
        How to combine per-sub-gene masks. 'any' (default) selects cells positive for at
        least one sub-gene; 'all' requires positivity in every sub-gene.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, optional
        GMM covariance structure used for both the sub-gene and marker fits (default 'full').

    Returns
    -------
    dict with keys:
        ``marker_gene``         – name of the marker gene
        ``subgene_thresholds``  – dict mapping each sub-gene to its threshold dict
        ``subgene_mask``        – boolean pd.Series aligned to pivot_df index
        ``n_masked_cells``      – number of cells passing the sub-gene mask
        ``threshold_log2``      – marker threshold in log2(count + 1) space
        ``threshold_raw``       – marker threshold back-transformed to raw count space
        ``gmm``                 – fitted GaussianMixture object on the masked population
        ``log_values``          – log2-transformed marker values used for the refit
        ``covariance_type``     – covariance type used for all fits
    """
    if mask_logic not in ("any", "all"):
        raise ValueError("mask_logic must be 'any' or 'all'.")

    # --- Step 1 & 2: threshold each sub-gene and build per-gene boolean mask ----------------
    subgene_thresholds = {}
    per_gene_masks = {}
    for sg in subgenes:
        result = fit_gmm_threshold(
            pivot_df[sg].values,
            gene=sg,
            remove_zeros=remove_zeros,
            threshold_prob=threshold_prob,
            covariance_type=covariance_type,
        )
        subgene_thresholds[sg] = result
        per_gene_masks[sg] = pivot_df[sg] > result["threshold_raw"]

    # --- Step 3: combine masks --------------------------------------------------------------
    mask_df = pd.DataFrame(per_gene_masks, index=pivot_df.index)
    if mask_logic == "any":
        combined_mask = mask_df.any(axis=1)
    else:
        combined_mask = mask_df.all(axis=1)

    n_masked = int(combined_mask.sum())
    print(
        f"[GMM] '{marker_gene}' refit: {n_masked} / {len(pivot_df)} cells pass "
        f"sub-gene mask ({mask_logic.upper()} of {subgenes})."
    )
    if n_masked < 20:
        warnings.warn(
            f"[GMM] Only {n_masked} cells pass the sub-gene mask for '{marker_gene}'. "
            "GMM fit may be unreliable.",
            stacklevel=2,
        )

    # --- Step 4 & 5: extract marker counts for masked cells and refit -----------------------
    marker_values = pivot_df.loc[combined_mask, marker_gene].values
    refit_result = fit_gmm_threshold(
        marker_values,
        gene=f"{marker_gene} (sub-gene masked)",
        remove_zeros=remove_zeros,
        threshold_prob=threshold_prob,
        covariance_type=covariance_type,
    )

    return {
        "marker_gene": marker_gene,
        "subgene_thresholds": subgene_thresholds,
        "subgene_mask": combined_mask,
        "n_masked_cells": n_masked,
        "threshold_log2": refit_result["threshold_log2"],
        "threshold_raw": refit_result["threshold_raw"],
        "gmm": refit_result["gmm"],
        "log_values": refit_result["log_values"],
        "covariance_type": covariance_type,
    }


# -------------------------------------------------------------------------------------------------
# 8. QC plot: sub-gene-anchored refit
# -------------------------------------------------------------------------------------------------

@saveable_plot()
def plot_gmm_marker_subgene_analysis(
    pivot_df: pd.DataFrame,
    marker_gene: str,
    subgenes: list,
    remove_zeros: bool = True,
    threshold_prob: float = 0.5,
    mask_logic: str = "any",
    annotate_threshold: bool = True,
    covariance_type: str = "full",
) -> plt.Figure:
    """
    Two-panel QC figure for the sub-gene-anchored marker gene refit.

    Left panel  — horizontal bar chart showing how many cells each sub-gene contributes
                  to the positive mask, with an 'any positive' total bar for reference.
    Right panel — GMM fit on the masked marker-gene population (histogram + component
                  curves + mixture density + threshold line), via :func:`plot_gmm_fit`.

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Cell × gene pivot table with raw spot counts.
    marker_gene : str
        Broad marker gene to refit (e.g. 'Gad2', 'GFP').
    subgenes : list of str
        Sub-type genes used to build the positive-cell mask.
    remove_zeros : bool, optional
        Remove zeros before fitting (default True).
    threshold_prob : float, optional
        Posterior crossover probability (default 0.5).
    mask_logic : {'any', 'all'}, optional
        Mask combination logic (default 'any').
    annotate_threshold : bool, optional
        Annotate the threshold value on the GMM plot (default True).
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, optional
        GMM covariance structure (default 'full').

    Returns
    -------
    matplotlib.figure.Figure
    """
    result = fit_gmm_marker_via_subgenes(
        pivot_df,
        marker_gene=marker_gene,
        subgenes=subgenes,
        remove_zeros=remove_zeros,
        threshold_prob=threshold_prob,
        mask_logic=mask_logic,
        covariance_type=covariance_type,
    )

    fig, (ax_mask, ax_gmm) = plt.subplots(
        1, 2,
        figsize=(11, 4),
        gridspec_kw={"width_ratios": [1, 1.6]},
        constrained_layout=True,
    )

    # --- Left panel: per-sub-gene cell counts in mask ---------------------------------------
    per_gene_counts = {
        sg: int((pivot_df[sg] > result["subgene_thresholds"][sg]["threshold_raw"]).sum())
        for sg in subgenes
    }
    # add the combined total
    labels = list(per_gene_counts.keys()) + [f"any positive\n(mask total)"]
    counts = list(per_gene_counts.values()) + [result["n_masked_cells"]]
    bar_colors = ["#5B8DB8"] * len(subgenes) + ["#4A7A4A"]

    bars = ax_mask.barh(labels, counts, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax_mask.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=8,
        )
    ax_mask.set_xlabel("Number of positive cells")
    ax_mask.set_title(f"Sub-gene mask breakdown\n(threshold_prob={threshold_prob})", fontsize=9)
    ax_mask.invert_yaxis()
    ax_mask.spines[["top", "right"]].set_visible(False)

    # --- Right panel: GMM fit on masked marker population ----------------------------------
    plot_gmm_fit(
        ax_gmm,
        result["log_values"],
        gene=f"{marker_gene}  [cells positive for {mask_logic.upper()}({', '.join(subgenes)})]",
        gmm=result["gmm"],
        threshold_log2=result["threshold_log2"],
        annotate_threshold=annotate_threshold,
    )
    ax_gmm.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"{marker_gene} GMM refit anchored by sub-gene mask  "
        f"(n={result['n_masked_cells']} cells)",
        fontsize=10,
    )
    return fig


# -------------------------------------------------------------------------------------------------
# 9. End-to-end inhibitory gene thresholding pipeline
# -------------------------------------------------------------------------------------------------

_DEFAULT_ANCHOR_GENES = ("Pvalb", "Sst", "Npy","Vip")


def run_inhibitory_gmm_thresholding(
    pivot_df: pd.DataFrame,
    inh_genes: list,
    marker_gene: str = "Gad2",
    anchor_genes: tuple = _DEFAULT_ANCHOR_GENES,
    min_count: int = 5,
    covariance_type: str = "tied",
    threshold_prob: float = 0.5,
    plot_qc: bool = True,
    annotate_threshold: bool = True,
    grid_save_kwargs: dict = None,
    refit_save_kwargs: dict = None,
) -> tuple:
    """
    End-to-end GMM thresholding pipeline for a panel of inhibitory marker genes.

    Steps
    -----
    1. Zero out counts below ``min_count`` (noise floor filter).
    2. Fit GMM thresholds for all genes in ``inh_genes`` via :func:`threshold_genes`.
       If ``marker_gene`` is present its row is labelled ``source="gmm_no_refit"``
       to distinguish it from the anchored refit that follows.
    3. Optionally plot a QC grid of all fits (saveable via ``grid_save_kwargs``).
    4. Refit ``marker_gene`` restricted to cells positive for any of ``anchor_genes``
       via :func:`fit_gmm_marker_via_subgenes`.  The resulting threshold *replaces* the
       ``gmm_no_refit`` row; the original is retained as ``source="gmm_no_refit"`` for
       comparison.
    5. Optionally plot the two-panel refit QC figure (saveable via ``refit_save_kwargs``).
    6. Return the combined thresholds DataFrame and the noise-floor-filtered pivot.

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Cell × gene pivot table with raw spot counts.
    inh_genes : list of str
        Inhibitory marker genes to threshold (e.g. ``["GFP","Gad2","Pvalb","Sst","Vip","Npy"]``).
    marker_gene : str, optional
        Gene to apply the sub-gene-anchored refit to (default ``"Gad2"``).
        Must be present in ``inh_genes``.  Set to ``None`` to skip the refit step.
    anchor_genes : tuple of str, optional
        Sub-type genes used to build the positive-cell mask for the refit
        (default ``("Pvalb", "Sst", "Vip", "Npy")``).
    min_count : int, optional
        Counts strictly below this value are set to zero before fitting (default 5).
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, optional
        GMM covariance structure for all fits (default ``'tied'``).
    threshold_prob : float, optional
        Posterior crossover probability (default 0.5).
    plot_qc : bool, optional
        Whether to generate QC plots (default True).
    annotate_threshold : bool, optional
        Annotate threshold values on QC plots (default True).
    grid_save_kwargs : dict, optional
        Keyword arguments forwarded to :func:`plot_gmm_threshold_grid` for saving.
        Example: ``{"save": True, "output_dir": "/results", "filename": "gmm_grid"}``.
        All keys accepted by the ``@saveable_plot`` decorator are valid.
    refit_save_kwargs : dict, optional
        Keyword arguments forwarded to :func:`plot_gmm_marker_subgene_analysis` for saving.
        Example: ``{"save": True, "output_dir": "/results", "filename": "gmm_gad2_refit"}``.

    Returns
    -------
    combined_thresholds_df : pd.DataFrame
        Columns: ``gene``, ``threshold_log2``, ``threshold_raw``, ``source``.
        ``source`` values:
        - ``"gmm_<covariance_type>"``        — standard fit for most genes
        - ``"gmm_no_refit"``                 — standard fit for ``marker_gene`` (kept for comparison)
        - ``"gmm_subgene_refit"``            — sub-gene-anchored refit for ``marker_gene``
    pivot_filtered : pd.DataFrame
        Copy of ``pivot_df`` with counts < ``min_count`` zeroed out.
    """
    grid_save_kwargs = grid_save_kwargs or {}
    refit_save_kwargs = refit_save_kwargs or {}

    # --- Step 1: noise floor filter ---------------------------------------------------------
    pivot_filtered = pivot_df.where(pivot_df >= min_count, other=0)
    print(f"[GMM] Noise floor: counts < {min_count} set to 0.")

    # --- Step 2: standard threshold for all inh_genes --------------------------------------
    marker_present = marker_gene is not None and marker_gene in inh_genes
    source_label = f"gmm_{covariance_type}"

    raw_thresholds = threshold_genes(
        pivot_filtered,
        genes=inh_genes,
        covariance_type=covariance_type,
        threshold_prob=threshold_prob,
    )
    raw_thresholds["source"] = source_label
    if marker_present:
        raw_thresholds.loc[raw_thresholds["gene"] == marker_gene, "source"] = "gmm_no_refit"

    # --- Step 3: optional QC grid -----------------------------------------------------------
    if plot_qc:
        fig_grid = plot_gmm_threshold_grid(
            pivot_filtered,
            genes=inh_genes,
            covariance_type=covariance_type,
            threshold_prob=threshold_prob,
            annotate_threshold=annotate_threshold,
            **grid_save_kwargs,
        )
        plt.show()

    # --- Step 4: sub-gene-anchored refit for marker_gene -----------------------------------
    refit_rows = []
    if marker_present:
        # Only use anchor genes that are actually present in the pivot
        valid_anchors = [g for g in anchor_genes if g in pivot_filtered.columns]
        if len(valid_anchors) < len(anchor_genes):
            missing = set(anchor_genes) - set(valid_anchors)
            warnings.warn(f"[GMM] Anchor genes not found in pivot and skipped: {missing}", stacklevel=2)

        refit = fit_gmm_marker_via_subgenes(
            pivot_filtered,
            marker_gene=marker_gene,
            subgenes=valid_anchors,
            covariance_type=covariance_type,
            threshold_prob=threshold_prob,
        )
        refit_rows.append({
            "gene": marker_gene,
            "threshold_log2": refit["threshold_log2"],
            "threshold_raw": refit["threshold_raw"],
            "source": "gmm_subgene_refit",
        })

        # --- Step 5: optional refit QC plot -----------------------------------------------
        if plot_qc:
            fig_refit = plot_gmm_marker_subgene_analysis(
                pivot_filtered,
                marker_gene=marker_gene,
                subgenes=valid_anchors,
                covariance_type=covariance_type,
                threshold_prob=threshold_prob,
                annotate_threshold=annotate_threshold,
                **refit_save_kwargs,
            )
            plt.show()

    # --- Step 6: combine -------------------------------------------------------------------
    combined_thresholds_df = pd.concat(
        [raw_thresholds, pd.DataFrame(refit_rows)],
        ignore_index=True,
    )
    print(
        f"[GMM] Pipeline complete. {len(combined_thresholds_df)} threshold rows "
        f"for genes: {combined_thresholds_df['gene'].tolist()}."
    )
    return combined_thresholds_df, pivot_filtered


# -------------------------------------------------------------------------------------------------
# 10. Filter cells by GMM thresholds
# -------------------------------------------------------------------------------------------------


def filter_cells_by_gmm_thresholds(
    pivot_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    logic: str = "any",
    source_priority: str = "gmm_subgene_refit",
) -> pd.DataFrame:
    """
    Return a cell × gene pivot restricted to cells that exceed at least one
    (or all) GMM-derived thresholds.

    When ``thresholds_df`` contains multiple rows for the same gene (e.g. both
    ``"gmm_no_refit"`` and ``"gmm_subgene_refit"`` for Gad2), the row whose
    ``source`` matches ``source_priority`` is used; if no such row exists for a
    gene, the last row for that gene is used.

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Cell × gene pivot table with raw spot counts (cells as rows, genes as columns).
    thresholds_df : pd.DataFrame
        Output of :func:`run_inhibitory_gmm_thresholding` or :func:`threshold_genes`.
        Must contain columns ``gene`` and ``threshold_raw``.
    logic : {'any', 'all'}, optional
        ``'any'`` (default) keeps cells positive for at least one gene.
        ``'all'`` keeps cells positive for every gene.
    source_priority : str, optional
        When multiple rows exist for a gene, prefer this ``source`` value.
        Default ``"gmm_subgene_refit"`` ensures the anchored Gad2 threshold is used
        over the ``"gmm_no_refit"`` row when both are present.

    Returns
    -------
    pd.DataFrame
        Filtered pivot containing only cells that pass the threshold gate.
    """
    if logic not in ("any", "all"):
        raise ValueError("logic must be 'any' or 'all'.")

    # Resolve one threshold per gene, preferring source_priority where available
    resolved = {}
    for gene, group in thresholds_df.groupby("gene"):
        if gene not in pivot_df.columns:
            continue
        priority_rows = group[group["source"] == source_priority] if "source" in group.columns else pd.DataFrame()
        row = priority_rows.iloc[0] if not priority_rows.empty else group.iloc[-1]
        resolved[gene] = float(row["threshold_raw"])

    if not resolved:
        raise ValueError("No genes in thresholds_df matched columns in pivot_df.")

    # Build per-gene boolean masks
    per_gene_masks = {gene: pivot_df[gene] > thr for gene, thr in resolved.items()}
    mask_df = pd.DataFrame(per_gene_masks, index=pivot_df.index)

    if logic == "any":
        cell_mask = mask_df.any(axis=1)
    else:
        cell_mask = mask_df.all(axis=1)

    n_pass = int(cell_mask.sum())
    print(
        f"[GMM] filter_cells_by_gmm_thresholds: {n_pass} / {len(pivot_df)} cells pass "
        f"({logic.upper()} of {list(resolved.keys())})."
    )
    return pivot_df.loc[cell_mask].copy()

