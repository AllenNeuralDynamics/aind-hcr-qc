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
        "centroids_a": centroids_a,
        "centroids_b": centroids_b,
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


def plot_matched_centroid_heatmap(
    result: dict,
    label_a: str = "Version A",
    label_b: str = "Version B",
    gene_order: Optional[list[str]] = None,
    normalize_genes: bool = True,
    cmap: str = "YlOrRd",
    similarity_threshold: float = 0.8,
    figsize: tuple[float, float] = (18, 10),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Matched centroid heatmap with per-pair cluster-size bars and similarity strip.

    Each matched cluster pair occupies two adjacent rows — the top row is
    dataset A's centroid, the bottom row is dataset B's — separated by a thin
    white line.  Three panels are arranged side by side:

    * **Left**: gene expression heatmap (genes × matched pairs × 2 sub-rows).
    * **Centre**: horizontal bar chart of cluster sizes (A and B side by side).
    * **Right**: coloured similarity strip (green ≥ threshold, red < threshold),
      with the numeric score annotated.

    Pairs are sorted best-match first (descending similarity).

    Parameters
    ----------
    result : dict
        Output of :func:`compute_cluster_similarity`.  Must contain
        ``centroids_a``, ``centroids_b``, ``ids_a``, ``ids_b``,
        ``sizes_a``, ``sizes_b``, ``match_df``, and ``gene_columns``.
    label_a, label_b : str
        Human-readable dataset names.
    gene_order : list of str, optional
        Explicit gene column order.  Defaults to sorting genes by their mean
        expression across all centroids (highest on the left).
    normalize_genes : bool
        If ``True`` (default), each gene column is independently scaled to
        ``[0, 1]`` so weak markers are not washed out by strong ones.
        If ``False``, raw centroid values are used and ``vmax`` applies.
    cmap : str
        Matplotlib colormap for the expression heatmap.
    similarity_threshold : float
        Scores at or above this value are coloured green in the similarity strip;
        below are coloured red.
    figsize : tuple
        Overall figure size ``(width, height)``.
    vmin : float, optional
        Lower clip for the colour scale.  When ``normalize_genes=True``,
        defaults to ``0.0``; when ``False``, defaults to ``0.0`` unless
        overridden.
    vmax : float, optional
        Upper clip for the colour scale when ``normalize_genes=False``.
        Defaults to the 99th-percentile of all centroid values.
        Ignored when ``normalize_genes=True``.

    Returns
    -------
    plt.Figure
    """
    match_df = result["match_df"].copy().sort_values("similarity", ascending=False).reset_index(drop=True)
    gene_columns = result["gene_columns"]
    centroids_a = result["centroids_a"]   # shape (n_a, n_genes)
    centroids_b = result["centroids_b"]   # shape (n_b, n_genes)
    ids_a = result["ids_a"]
    ids_b = result["ids_b"]
    sizes_a = result["sizes_a"]
    sizes_b = result["sizes_b"]

    # index lookup: cluster id → row in centroids array
    id_to_idx_a = {k: i for i, k in enumerate(ids_a)}
    id_to_idx_b = {k: i for i, k in enumerate(ids_b)}
    id_to_size_a = {k: s for k, s in zip(ids_a, sizes_a)}
    id_to_size_b = {k: s for k, s in zip(ids_b, sizes_b)}

    n_pairs = len(match_df)
    n_rows = n_pairs * 3 - 1   # two data rows + one NaN spacer per pair, no trailing spacer

    # --- build interleaved expression matrix ---
    # row order: [pair0_A, pair0_B, spacer, pair1_A, pair1_B, spacer, ..., pairN_A, pairN_B]
    gene_idx = {g: i for i, g in enumerate(gene_columns)}

    # determine gene order
    if gene_order is not None:
        ordered_genes = [g for g in gene_order if g in gene_idx]
    else:
        all_centroids = np.vstack([centroids_a, centroids_b])
        mean_expr = all_centroids.mean(axis=0)
        ordered_genes = [gene_columns[i] for i in np.argsort(-mean_expr)]

    col_idx = [gene_idx[g] for g in ordered_genes]

    expr_matrix = np.full((n_rows, len(ordered_genes)), np.nan)
    row_labels = [""] * n_rows          # pre-fill; spacer rows stay blank
    data_row_positions = []             # (y_in_matrix, pair_idx) for data rows
    pair_centre_y = []                  # y midpoint of each pair (for bar/strip alignment)

    for pair_idx, row in match_df.iterrows():
        ca = centroids_a[id_to_idx_a[row["cluster_a"]]]
        cb = centroids_b[id_to_idx_b[row["cluster_b"]]]
        r_a = pair_idx * 3       # A sub-row
        r_b = pair_idx * 3 + 1  # B sub-row
        # r_spacer = pair_idx * 3 + 2  (left as NaN)
        expr_matrix[r_a] = ca[col_idx]
        expr_matrix[r_b] = cb[col_idx]
        row_labels[r_a] = f"{int(row['cluster_a'])}  ({label_a})"
        row_labels[r_b] = f"{int(row['cluster_b'])}  ({label_b})"
        data_row_positions.append((r_a, r_b))
        pair_centre_y.append((r_a + r_b) / 2.0)

    # --- normalisation ---
    if normalize_genes:
        # per-column (gene) min-max scaling to [0, 1]
        data_only = expr_matrix[~np.all(np.isnan(expr_matrix), axis=1)]
        col_min = np.nanmin(data_only, axis=0)
        col_max = np.nanmax(data_only, axis=0)
        col_range = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
        display_matrix = (expr_matrix - col_min) / col_range   # NaN rows stay NaN
        _vmin = vmin if vmin is not None else 0.0
        _vmax = vmax if vmax is not None else 1.0
        cbar_label = "Normalised expression (per gene)"
    else:
        display_matrix = expr_matrix
        _vmin = vmin if vmin is not None else 0.0
        _vmax = vmax if vmax is not None else float(np.nanpercentile(expr_matrix[~np.isnan(expr_matrix)], 99)) or 1.0
        cbar_label = "Expression"

    # --- figure layout ---
    # widths: [heatmap, size_bars, similarity_strip]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[12, 3, 1],
        wspace=0.05,
    )
    ax_hm   = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])
    ax_sim  = fig.add_subplot(gs[2])

    # ── heatmap ──────────────────────────────────────────────────────────────
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")     # NaN spacer rows render as white

    im = ax_hm.imshow(
        display_matrix,
        aspect="auto",
        cmap=cmap_obj,
        vmin=_vmin,
        vmax=_vmax,
        interpolation="nearest",
    )
    ax_hm.set_xticks(range(len(ordered_genes)))
    ax_hm.set_xticklabels(ordered_genes, rotation=0, ha="center", fontsize=10)
    ax_hm.set_yticks([r for r_a, r_b in data_row_positions for r in (r_a, r_b)])
    ax_hm.set_yticklabels(
        [lb for r_a, r_b in data_row_positions for lb in (row_labels[r_a], row_labels[r_b])],
        fontsize=10,
    )
    ax_hm.set_title(f"Matched Cluster Centroids  ({label_a}  vs  {label_b})", fontsize=13)

    # colorbar
    cbar = fig.colorbar(im, ax=ax_hm, shrink=0.4, pad=0.01)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ── size bar chart ────────────────────────────────────────────────────────
    sizes_A = [id_to_size_a[row["cluster_a"]] for _, row in match_df.iterrows()]
    sizes_B = [id_to_size_b[row["cluster_b"]] for _, row in match_df.iterrows()]

    # each pair spans 3 row units (A=+0.5, B=-0.5 from centre); bars fill one row each
    bar_height = 0.9
    ax_bar.barh(
        [cy + 0.5 for cy in pair_centre_y], sizes_A,
        height=bar_height, color="#5B9BD5", label=label_a, align="center",
    )
    ax_bar.barh(
        [cy - 0.5 for cy in pair_centre_y], sizes_B,
        height=bar_height, color="#ED7D31", label=label_b, align="center",
    )
    ax_bar.set_ylim(-0.5, n_rows - 0.5)
    ax_bar.invert_yaxis()
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Cluster size (cells)", fontsize=10)
    ax_bar.set_title("Size", fontsize=11)
    ax_bar.tick_params(axis="x", labelsize=9)
    ax_bar.legend(fontsize=9, loc="lower right")
    ax_bar.spines[["top", "right", "left"]].set_visible(False)

    # ── similarity strip ─────────────────────────────────────────────────────
    similarities = match_df["similarity"].values

    for cy, sim in zip(pair_centre_y, similarities):
        color = (0.18, 0.63, 0.18, 1.0) if sim >= similarity_threshold else (0.80, 0.18, 0.18, 1.0)
        ax_sim.barh(cy, 1, color=color, height=2.0)
        ax_sim.text(
            0.5, cy, f"{sim:.2f}",
            ha="center", va="center",
            fontsize=9, color="white", fontweight="bold",
        )

    ax_sim.set_ylim(-0.5, n_rows - 0.5)
    ax_sim.invert_yaxis()
    ax_sim.set_xlim(0, 1)
    ax_sim.set_xticks([])
    ax_sim.set_yticks([])
    ax_sim.set_title("Sim.", fontsize=11)
    ax_sim.spines[["top", "right", "bottom", "left"]].set_visible(False)

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


# =================================================================================================
# Comparisons across cluster pairs analysis
#
# Functions for running all pairwise comparisons across a collection of mice/conditions
# and summarising the variability in cluster similarity scores.
#
# Workflow:
#   1. run_pairwise_comparisons        — run compute_cluster_similarity for every (A, B) pair
#   2. summarise_pairwise_comparisons  — collapse each comparison to mean/median/match_rate
#   3. plot_similarity_matrix_heatmap  — N×N heatmap of summary scores
#   4. plot_similarity_swarm           — swarm/strip plot of per-cluster scores per comparison
#   5. plot_per_cluster_variability    — bar chart of per-cluster std across all comparisons
#   6. plot_pairwise_comparison_summary — combined figure with all three panels
# =================================================================================================


def run_pairwise_comparisons(
    mouse_data: dict,
    metric: str = "pearson",
    use_size_weights: bool = False,
    log_transform: bool = True,
    clip_range: Optional[tuple[float, float]] = None,
    scale: bool = True,
) -> dict:
    """
    Run :func:`compute_cluster_similarity` for every unique pair of mice/conditions.

    Parameters
    ----------
    mouse_data : dict
        Mapping of ``label -> (cxg_pivot, cluster_labels)``.
        Example::

            {
                "767022": (final_cxg_767022, cluster_labels_767022),
                "755252": (final_cxg_755252, cluster_labels_755252),
            }

    metric : str
        Similarity metric passed to :func:`compute_cluster_similarity`.
    use_size_weights : bool
        Weight matching by cluster-size concordance.
    log_transform : bool
        Apply ``log1p`` before computing centroids.
    clip_range : tuple, optional
        Clip expression values before optional log-transform.
    scale : bool
        Z-score genes across all centroids before cosine similarity.

    Returns
    -------
    dict
        Keys are ``"label_a vs label_b"`` strings; values are the result dicts
        from :func:`compute_cluster_similarity`, each augmented with
        ``"label_a"`` and ``"label_b"`` string fields.
    """
    labels = list(mouse_data.keys())
    comparison_results = {}

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label_a, label_b = labels[i], labels[j]
            cxg_a, labs_a = mouse_data[label_a]
            cxg_b, labs_b = mouse_data[label_b]

            print(f"  Running: {label_a}  vs  {label_b}")
            result = compute_cluster_similarity(
                cxg_a=cxg_a,
                labels_a=labs_a,
                cxg_b=cxg_b,
                labels_b=labs_b,
                metric=metric,
                use_size_weights=use_size_weights,
                log_transform=log_transform,
                clip_range=clip_range,
                scale=scale,
            )
            result["label_a"] = label_a
            result["label_b"] = label_b
            key = f"{label_a} vs {label_b}"
            comparison_results[key] = result

    print(f"\nCompleted {len(comparison_results)} pairwise comparisons.")
    return comparison_results


def summarise_pairwise_comparisons(
    comparison_results: dict,
    similarity_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Collapse each pairwise comparison to summary statistics.

    Parameters
    ----------
    comparison_results : dict
        Output of :func:`run_pairwise_comparisons`.
    similarity_threshold : float
        A cluster match is "good" if its similarity >= this value.

    Returns
    -------
    pd.DataFrame
        One row per comparison.  Columns:

        ``comparison``       — "label_a vs label_b"
        ``label_a``          — mouse/condition A
        ``label_b``          — mouse/condition B
        ``mean_similarity``  — mean across all matched cluster pairs
        ``median_similarity``— median across all matched cluster pairs
        ``std_similarity``   — std across all matched cluster pairs
        ``match_rate``       — fraction of pairs with similarity >= threshold
        ``n_clusters``       — number of matched pairs
    """
    rows = []
    for key, result in comparison_results.items():
        scores = result["match_df"]["similarity"].values
        rows.append({
            "comparison": key,
            "label_a": result["label_a"],
            "label_b": result["label_b"],
            "mean_similarity": float(scores.mean()),
            "median_similarity": float(np.median(scores)),
            "std_similarity": float(scores.std()),
            "match_rate": float((scores >= similarity_threshold).mean()),
            "n_clusters": len(scores),
        })
    return pd.DataFrame(rows).sort_values("mean_similarity", ascending=False).reset_index(drop=True)


def _build_all_scores_df(comparison_results: dict) -> pd.DataFrame:
    """
    Flatten all per-cluster similarity scores into a single long-form DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: comparison, label_a, label_b, cluster_a, cluster_b, similarity
    """
    rows = []
    for key, result in comparison_results.items():
        for _, r in result["match_df"].iterrows():
            rows.append({
                "comparison": key,
                "label_a": result["label_a"],
                "label_b": result["label_b"],
                "cluster_a": r["cluster_a"],
                "cluster_b": r["cluster_b"],
                "similarity": r["similarity"],
            })
    return pd.DataFrame(rows)


def plot_similarity_matrix_heatmap(
    summary_df: pd.DataFrame,
    score_col: str = "mean_similarity",
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (7, 6),
    cmap: str = "YlGnBu",
    vmin: float = 0.0,
    vmax: float = 1.0,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Symmetric N×N heatmap of pairwise mean similarity scores.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of :func:`summarise_pairwise_comparisons`.
    score_col : str
        Column from *summary_df* to use as the cell value
        (e.g. ``'mean_similarity'``, ``'match_rate'``).
    ax : plt.Axes, optional
    figsize : tuple
    cmap : str
    vmin, vmax : float
        Colour scale limits.
    title : str, optional

    Returns
    -------
    plt.Figure
    """
    # Build symmetric matrix
    all_labels = sorted(set(summary_df["label_a"]) | set(summary_df["label_b"]))
    n = len(all_labels)
    idx = {l: i for i, l in enumerate(all_labels)}
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)

    for _, row in summary_df.iterrows():
        i, j = idx[row["label_a"]], idx[row["label_b"]]
        mat[i, j] = row[score_col]
        mat[j, i] = row[score_col]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=all_labels,
        yticklabels=all_labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(title or f"Pairwise {score_col.replace('_', ' ').title()}", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def plot_similarity_swarm(
    comparison_results: dict,
    similarity_threshold: float = 0.8,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (10, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Strip/swarm plot of per-cluster similarity scores, one column per comparison.

    Each dot is one matched cluster pair.  A box overlay shows the median and
    IQR.  A dashed horizontal line marks the similarity threshold.

    Parameters
    ----------
    comparison_results : dict
        Output of :func:`run_pairwise_comparisons`.
    similarity_threshold : float
        Dashed reference line position.
    ax : plt.Axes, optional
    figsize : tuple
    title : str, optional

    Returns
    -------
    plt.Figure
    """
    scores_df = _build_all_scores_df(comparison_results)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    comparison_order = list(comparison_results.keys())

    sns.boxplot(
        data=scores_df,
        x="comparison",
        y="similarity",
        order=comparison_order,
        color="lightgrey",
        width=0.4,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=scores_df,
        x="comparison",
        y="similarity",
        order=comparison_order,
        jitter=True,
        size=6,
        alpha=0.7,
        palette="tab10",
        hue="comparison",
        legend=False,
        ax=ax,
    )
    ax.axhline(similarity_threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"threshold = {similarity_threshold}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Cluster similarity score")
    ax.set_title(title or "Per-cluster similarity scores per pairwise comparison", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_per_cluster_variability(
    comparison_results: dict,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (10, 4),
    title: Optional[str] = None,
    color_threshold: float = 0.15,
) -> plt.Figure:
    """
    Bar chart of per-cluster-A std of similarity across all comparisons.

    A high std for a cluster means its match quality varies a lot across
    mouse pairs — it is an unstable cluster type.  A low std means it
    matches consistently well (or consistently poorly).

    Parameters
    ----------
    comparison_results : dict
        Output of :func:`run_pairwise_comparisons`.
    ax : plt.Axes, optional
    figsize : tuple
    title : str, optional
    color_threshold : float
        Bars with std >= this value are coloured orange (high variability);
        below are blue (stable).

    Returns
    -------
    plt.Figure
    """
    scores_df = _build_all_scores_df(comparison_results)

    # group by cluster_a — std of similarity across comparisons it appears in
    cluster_stats = (
        scores_df.groupby("cluster_a")["similarity"]
        .agg(mean_sim="mean", std_sim="std", n_comparisons="count")
        .reset_index()
        .sort_values("cluster_a")
    )
    cluster_stats["std_sim"] = cluster_stats["std_sim"].fillna(0)

    colors = [
        "#FF9800" if s >= color_threshold else "#5B9BD5"
        for s in cluster_stats["std_sim"]
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bars = ax.bar(
        cluster_stats["cluster_a"].astype(str),
        cluster_stats["std_sim"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(color_threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"variability threshold = {color_threshold}")

    # annotate mean similarity above each bar
    for bar, mean_val in zip(bars, cluster_stats["mean_sim"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean_val:.2f}",
            ha="center", va="bottom", fontsize=7, color="grey",
        )

    ax.set_xlabel("Cluster (A label)")
    ax.set_ylabel("Std of similarity\nacross comparisons")
    ax.set_title(title or "Per-cluster variability across pairwise comparisons\n(annotated with mean similarity)",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(cluster_stats["std_sim"].max() * 1.3, color_threshold * 1.5))
    fig.tight_layout()
    return fig


def plot_pairwise_comparison_summary(
    comparison_results: dict,
    similarity_threshold: float = 0.8,
    score_col: str = "mean_similarity",
    figsize: tuple[float, float] = (20, 14),
    vmin: float = 0.0,
    vmax: float = 1.0,
    color_threshold: float = 0.15,
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Combined three-panel summary figure for multi-mouse pairwise comparisons.

    Panels
    ------
    Top-left  : N×N heatmap of pairwise mean similarity.
    Top-right : Swarm + box plot of per-cluster scores per comparison.
    Bottom    : Per-cluster std bar chart with mean similarity annotations.

    Parameters
    ----------
    comparison_results : dict
        Output of :func:`run_pairwise_comparisons`.
    similarity_threshold : float
        Reference line in the swarm plot.
    score_col : str
        Column to use in the N×N heatmap
        (``'mean_similarity'``, ``'median_similarity'``, or ``'match_rate'``).
    figsize : tuple
        Overall figure size.
    vmin, vmax : float
        Colour scale for the N×N heatmap.
    color_threshold : float
        Std threshold for colouring bars in the variability chart.

    Returns
    -------
    fig : plt.Figure
    summary_df : pd.DataFrame
        Output of :func:`summarise_pairwise_comparisons`.
    """
    summary_df = summarise_pairwise_comparisons(
        comparison_results,
        similarity_threshold=similarity_threshold,
    )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          height_ratios=[1.2, 1])

    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_swarm   = fig.add_subplot(gs[0, 1])
    ax_var     = fig.add_subplot(gs[1, :])

    plot_similarity_matrix_heatmap(
        summary_df,
        score_col=score_col,
        ax=ax_heatmap,
        vmin=vmin,
        vmax=vmax,
    )

    plot_similarity_swarm(
        comparison_results,
        similarity_threshold=similarity_threshold,
        ax=ax_swarm,
    )

    plot_per_cluster_variability(
        comparison_results,
        ax=ax_var,
        color_threshold=color_threshold,
    )

    fig.suptitle(
        "Pairwise cluster similarity summary across mice/conditions",
        fontsize=14, y=1.01,
    )
    return fig, summary_df
