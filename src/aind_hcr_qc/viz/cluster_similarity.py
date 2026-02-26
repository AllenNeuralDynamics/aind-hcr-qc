"""
Cluster similarity comparison for cell x gene tables.

Compares KMeans (or other) cluster versions across different processing runs
by computing centroid similarity and finding optimal cluster matches.
"""

from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------------------------------------
# Processing
# -------------------------------------------------------------------------------------------------


def _to_pivot(cxg: pd.DataFrame) -> pd.DataFrame:
    """
    Detect whether *cxg* is long-form and pivot it if necessary.

    Long-form is identified by the presence of a ``gene`` column together with
    a ``spot_count`` (or ``count``) column.  Any other shape is assumed to
    already be a cells x genes matrix (index = cell_id, columns = genes).

    Long-form expected columns: cell_id, round_key, chan, gene, spot_count

    Parameters
    ----------
    cxg : pd.DataFrame
        Either a long-form table with columns
        ``[cell_id, gene, spot_count, ...]`` or a pivot table
        ``(cell_id x genes)``.

    Returns
    -------
    pd.DataFrame
        Pivot table with ``cell_id`` as index and genes as columns.
    """
    if "gene" in cxg.columns:
        value_col = "spot_count" if "spot_count" in cxg.columns else "count"
        pivot = (
            cxg.pivot_table(index="cell_id", columns="gene", values=value_col, aggfunc="sum")
            .fillna(0)
        )
        pivot.columns.name = None
        return pivot
    # already a pivot — ensure numeric only
    return cxg.select_dtypes(include="number")


def prepare_expression_matrix(
    cxg: pd.DataFrame,
    log_transform: bool = False,
    clip_range: Optional[tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Convert a cell x gene table to a cleaned numeric matrix, with optional
    clipping and log-transform.

    Parameters
    ----------
    cxg : pd.DataFrame
        Long-form or pivot cell x gene table.
    log_transform : bool
        If True, apply ``log1p`` to expression values.
    clip_range : tuple of (min, max), optional
        If provided, clip values to this range *before* log-transform.

    Returns
    -------
    pd.DataFrame
        Cells x genes float matrix.
    """
    mat = _to_pivot(cxg).astype(float)
    if clip_range is not None:
        mat = mat.clip(lower=clip_range[0], upper=clip_range[1])
    if log_transform:
        mat = np.log1p(mat)
    return mat


def compute_cluster_centroids(
    expr_matrix: pd.DataFrame,
    labels: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-cluster mean centroids from a cells x genes expression matrix.

    Parameters
    ----------
    expr_matrix : pd.DataFrame
        Cells x genes matrix (index = cell_id, columns = genes).
    labels : array-like
        Cluster label for each row of *expr_matrix*, aligned by position.

    Returns
    -------
    centroids : np.ndarray, shape (n_clusters, n_genes)
    sizes : np.ndarray, shape (n_clusters,)
        Number of cells per cluster.
    cluster_ids : np.ndarray, shape (n_clusters,)
        Ordered unique cluster labels.
    """
    labels = np.asarray(labels)
    cluster_ids = np.unique(labels)
    X = expr_matrix.values
    centroids = np.array([X[labels == k].mean(axis=0) for k in cluster_ids])
    sizes = np.array([(labels == k).sum() for k in cluster_ids])
    return centroids, sizes, cluster_ids


def compute_similarity_matrix(
    centroids_a: np.ndarray,
    centroids_b: np.ndarray,
    metric: str = "cosine",
    scale: bool = False,
) -> np.ndarray:
    """
    Compute pairwise similarity between two sets of cluster centroids.

    Parameters
    ----------
    centroids_a : np.ndarray, shape (n_a, n_genes)
    centroids_b : np.ndarray, shape (n_b, n_genes)
    metric : {'cosine', 'pearson'}
        Similarity metric to use.
    scale : bool
        If True, z-score each gene across all centroids (A and B jointly) via
        ``StandardScaler`` before computing cosine similarity.  Has no effect
        when ``metric='pearson'`` (pearson already z-scores per centroid row).

    Returns
    -------
    np.ndarray, shape (n_a, n_b)
    """
    if metric == "cosine":
        if scale:
            all_centroids = np.vstack([centroids_a, centroids_b])
            scaler = StandardScaler()
            all_scaled = scaler.fit_transform(all_centroids)
            centroids_a = all_scaled[:len(centroids_a)]
            centroids_b = all_scaled[len(centroids_a):]
        return cosine_similarity(centroids_a, centroids_b)
    elif metric == "pearson":
        def _zscore(X):
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, keepdims=True) + 1e-12
            return (X - mu) / sd
        za = _zscore(centroids_a)
        zb = _zscore(centroids_b)
        n = centroids_a.shape[1]
        return (za @ zb.T) / n
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose 'cosine' or 'pearson'.")


def apply_size_weights(
    sim_matrix: np.ndarray,
    sizes_a: np.ndarray,
    sizes_b: np.ndarray,
) -> np.ndarray:
    """
    Penalise matches between clusters of very different sizes.

    Weight = ``min(size_a, size_b) / max(size_a, size_b)`` (Jaccard-like),
    then rescaled so the range matches the raw similarity matrix.

    Parameters
    ----------
    sim_matrix : np.ndarray, shape (n_a, n_b)
    sizes_a : np.ndarray, shape (n_a,)
    sizes_b : np.ndarray, shape (n_b,)

    Returns
    -------
    np.ndarray, shape (n_a, n_b)
        Weighted similarity matrix.
    """
    outer_min = np.minimum.outer(sizes_a, sizes_b).astype(float)
    outer_max = np.maximum.outer(sizes_a, sizes_b).astype(float)
    size_concordance = outer_min / (outer_max + 1e-12)
    weighted = sim_matrix * size_concordance
    max_val = weighted.max()
    if max_val > 0:
        weighted = weighted / max_val * sim_matrix.max()
    return weighted


def match_clusters(
    sim_matrix: np.ndarray,
    ids_a: np.ndarray,
    ids_b: np.ndarray,
    sizes_a: np.ndarray,
    sizes_b: np.ndarray,
    weighted_sim_matrix: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Find the optimal one-to-one cluster mapping via the Hungarian algorithm.

    Parameters
    ----------
    sim_matrix : np.ndarray, shape (n_a, n_b)
        Raw similarity matrix (reported as ``similarity`` column).
    ids_a, ids_b : array-like
        Cluster labels for set A and B.
    sizes_a, sizes_b : array-like
        Cluster sizes for set A and B.
    weighted_sim_matrix : np.ndarray, optional
        If provided, Hungarian algorithm maximises this; otherwise uses
        *sim_matrix*.

    Returns
    -------
    pd.DataFrame
        Columns: cluster_a, cluster_b, size_a, size_b,
                 similarity, weighted_similarity
    """
    cost_matrix = weighted_sim_matrix if weighted_sim_matrix is not None else sim_matrix
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    records = []
    for r, c in zip(row_ind, col_ind):
        records.append({
            "cluster_a": ids_a[r],
            "cluster_b": ids_b[c],
            "size_a": int(sizes_a[r]),
            "size_b": int(sizes_b[c]),
            "similarity": float(sim_matrix[r, c]),
            "weighted_similarity": float(cost_matrix[r, c]),
        })
    return (
        pd.DataFrame(records)
        .sort_values("similarity", ascending=False)
        .reset_index(drop=True)
    )


def compute_cluster_similarity(
    cxg_a: pd.DataFrame,
    labels_a: np.ndarray | pd.Series,
    cxg_b: pd.DataFrame,
    labels_b: np.ndarray | pd.Series,
    metric: str = "cosine",
    use_size_weights: bool = False,
    log_transform: bool = False,
    clip_range: Optional[tuple[float, float]] = None,
    scale: bool = False,
) -> dict:
    """
    End-to-end processing: prepare matrices -> centroids -> similarity -> matches.

    Parameters
    ----------
    cxg_a, cxg_b : pd.DataFrame
        Long-form or pivot cell x gene tables.
        Long-form auto-detected via presence of ``gene`` + ``spot_count`` columns.
        Long-form columns: cell_id, round_key, chan, gene, spot_count
    labels_a, labels_b : array-like
        Cluster label per cell, one entry per *unique cell* (aligned to the
        unique cell order after pivoting).
    metric : {'cosine', 'pearson'}
    use_size_weights : bool
        Weight matching score by cluster-size concordance.
    log_transform : bool
        Apply ``log1p`` before computing centroids.
    clip_range : tuple of (min, max), optional
        Clip expression values before optional log-transform.
    scale : bool
        If True, z-score each gene across all centroids (A and B jointly) before
        computing cosine similarity.  Has no effect when ``metric='pearson'``.

    Returns
    -------
    dict with keys:
        sim_matrix, weighted_matrix, match_df,
        ids_a, ids_b, sizes_a, sizes_b, gene_columns
    """
    mat_a = prepare_expression_matrix(cxg_a, log_transform=log_transform, clip_range=clip_range)
    mat_b = prepare_expression_matrix(cxg_b, log_transform=log_transform, clip_range=clip_range)

    # align to shared genes
    shared_genes = sorted(set(mat_a.columns) & set(mat_b.columns))
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between the two cell x gene tables.")
    dropped_a = len(mat_a.columns) - len(shared_genes)
    dropped_b = len(mat_b.columns) - len(shared_genes)
    if dropped_a > 0 or dropped_b > 0:
        warnings.warn(
            f"Gene sets differ; using {len(shared_genes)} shared genes "
            f"(dropped {dropped_a} from A, {dropped_b} from B)."
        )
    mat_a = mat_a[shared_genes]
    mat_b = mat_b[shared_genes]

    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    centroids_a, sizes_a, ids_a = compute_cluster_centroids(mat_a, labels_a)
    centroids_b, sizes_b, ids_b = compute_cluster_centroids(mat_b, labels_b)

    sim_matrix = compute_similarity_matrix(centroids_a, centroids_b, metric=metric, scale=scale)

    weighted_matrix = None
    if use_size_weights:
        weighted_matrix = apply_size_weights(sim_matrix, sizes_a, sizes_b)

    match_df = match_clusters(
        sim_matrix, ids_a, ids_b, sizes_a, sizes_b,
        weighted_sim_matrix=weighted_matrix,
    )

    return {
        "sim_matrix": sim_matrix,
        "weighted_matrix": weighted_matrix,
        "match_df": match_df,
        "ids_a": ids_a,
        "ids_b": ids_b,
        "sizes_a": sizes_a,
        "sizes_b": sizes_b,
        "gene_columns": shared_genes,
    }


# -------------------------------------------------------------------------------------------------
# Visualisation
# -------------------------------------------------------------------------------------------------


def _cluster_tick_labels(ids: np.ndarray, sizes: np.ndarray) -> list[str]:
    """Build tick labels like 'k3 (n=412)'."""
    return [f"{k}  (n={s})" for k, s in zip(ids, sizes)]


def plot_similarity_heatmap(
    result: dict,
    label_a: str = "Version A",
    label_b: str = "Version B",
    show_matches: bool = True,
    use_weighted: bool = False,
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Heatmap of the full n_a x n_b cluster similarity matrix.

    Optimal matched pairs are highlighted with a white rectangle when
    ``show_matches=True``.

    Parameters
    ----------
    result : dict
        Output of :func:`compute_cluster_similarity`.
    label_a, label_b : str
        Axis labels for version A (rows) and B (columns).
    show_matches : bool
        Annotate optimal matched pairs with a white border.
    use_weighted : bool
        Plot the size-weighted matrix instead of the raw similarity.
    cmap : str
        Matplotlib colormap name.
    ax : plt.Axes, optional
    figsize : tuple

    Returns
    -------
    plt.Figure
    """
    matrix = (
        result["weighted_matrix"]
        if (use_weighted and result["weighted_matrix"] is not None)
        else result["sim_matrix"]
    )
    ids_a, ids_b = result["ids_a"], result["ids_b"]
    sizes_a, sizes_b = result["sizes_a"], result["sizes_b"]

    row_labels = _cluster_tick_labels(ids_a, sizes_a)
    col_labels = _cluster_tick_labels(ids_b, sizes_b)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0,
        vmax=1,
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=0.4,
        linecolor="white",
        ax=ax,
    )

    if show_matches:
        match_df = result["match_df"]
        id_to_row = {k: i for i, k in enumerate(ids_a)}
        id_to_col = {k: j for j, k in enumerate(ids_b)}
        for _, row in match_df.iterrows():
            r = id_to_row.get(row["cluster_a"])
            c = id_to_col.get(row["cluster_b"])
            if r is not None and c is not None:
                ax.add_patch(plt.Rectangle(
                        (c, r), 1, 1,
                        fill=False, edgecolor="red", lw=2.5, clip_on=False,
                    ))
    title = "Cluster Similarity" + (" (size-weighted)" if use_weighted else "")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(label_b, fontsize=10)
    ax.set_ylabel(label_a, fontsize=10)
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def plot_match_summary(
    result: dict,
    label_a: str = "Version A",
    label_b: str = "Version B",
    similarity_threshold: float = 0.8,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (7, 4),
) -> plt.Figure:
    """
    Horizontal bar chart of per-match similarity scores.

    Bars are green >= threshold, orange below.

    Parameters
    ----------
    result : dict
        Output of :func:`compute_cluster_similarity`.
    label_a, label_b : str
        Human-readable version names used in y-axis labels.
    similarity_threshold : float
        Score below which a match is flagged uncertain.
    ax : plt.Axes, optional
    figsize : tuple

    Returns
    -------
    plt.Figure
    """
    match_df = result["match_df"].copy()
    match_df["pair"] = (
        match_df["cluster_a"].astype(str) + f"  ({label_a})"
        + "  ↔  "
        + match_df["cluster_b"].astype(str) + f"  ({label_b})"
    )
    colors = [
        "#4CAF50" if s >= similarity_threshold else "#FF9800"
        for s in match_df["similarity"]
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.barh(match_df["pair"], match_df["similarity"], color=colors)
    ax.axvline(
        similarity_threshold, color="red", linestyle="--",
        linewidth=1, label=f"threshold = {similarity_threshold}",
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Similarity")
    ax.set_title("Optimal Cluster Matches")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_cluster_similarity(
    cxg_a: pd.DataFrame,
    labels_a: np.ndarray | pd.Series,
    cxg_b: pd.DataFrame,
    labels_b: np.ndarray | pd.Series,
    label_a: str = "Version A",
    label_b: str = "Version B",
    metric: str = "cosine",
    use_size_weights: bool = False,
    log_transform: bool = False,
    clip_range: Optional[tuple[float, float]] = None,
    similarity_threshold: float = 0.8,
    figsize: tuple[float, float] = (16, 6),
    scale: bool = False,
) -> tuple[dict, plt.Figure]:
    """
    Convenience function: full pipeline + side-by-side heatmap and match summary.

    Parameters
    ----------
    cxg_a, cxg_b : pd.DataFrame
        Long-form (cell_id, round_key, chan, gene, spot_count) or pivot
        cell x gene tables. Long-form is auto-detected.
    labels_a, labels_b : array-like
        Cluster label per cell, one per unique cell aligned to the pivoted
        row order.
    label_a, label_b : str
        Human-readable names for version A and B.
    metric : {'cosine', 'pearson'}
    use_size_weights : bool
        Weight matching score by size concordance.
    log_transform : bool
        Apply ``log1p`` before computing centroids.
    clip_range : tuple of (min, max), optional
        Clip expression values before optional log-transform.
    similarity_threshold : float
        Threshold line in the match-summary bar chart.
    figsize : tuple
        Total figure size for the 1x2 layout.
    scale : bool
        If True, z-score each gene across all centroids (A and B jointly) via
        ``StandardScaler`` before computing cosine similarity.
        Has no effect when ``metric='pearson'``.

    Returns
    -------
    result : dict
        Full result dict from :func:`compute_cluster_similarity`.
    fig : plt.Figure
        Combined figure: heatmap (left) + match summary (right).
    """
    result = compute_cluster_similarity(
        cxg_a=cxg_a,
        labels_a=labels_a,
        cxg_b=cxg_b,
        labels_b=labels_b,
        metric=metric,
        use_size_weights=use_size_weights,
        log_transform=log_transform,
        clip_range=clip_range,
        scale=scale,
    )

    fig, (ax_hm, ax_bar) = plt.subplots(1, 2, figsize=figsize)

    plot_similarity_heatmap(
        result,
        label_a=label_a,
        label_b=label_b,
        show_matches=True,
        use_weighted=use_size_weights,
        ax=ax_hm,
    )
    plot_match_summary(
        result,
        label_a=label_a,
        label_b=label_b,
        similarity_threshold=similarity_threshold,
        ax=ax_bar,
    )

    title_parts = [f"Cluster Similarity  |  {label_a}  vs  {label_b}  |  metric={metric}"]
    if scale:
        title_parts.append("scaled")
    if use_size_weights:
        title_parts.append("size-weighted")
    if log_transform:
        title_parts.append("log1p")
    if clip_range:
        title_parts.append(f"clip={clip_range}")
    fig.suptitle("  |  ".join(title_parts), fontsize=12, y=1.01)
    fig.tight_layout()
    return result, fig
