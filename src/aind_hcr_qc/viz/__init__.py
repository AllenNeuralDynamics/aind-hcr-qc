from .camera_alignment import qc_camera_alignment
from .cell_x_gene import (
    plot_cell_x_gene_clustered,
    plot_cell_x_gene_simple,
    fig_mixed_unmixed_cxg_and_corr,
    # thresholding
    compute_gmm_bic_table,
    plot_bic_comparison,
    fit_gmm_threshold,
    threshold_genes,
    plot_gmm_threshold_grid,
    fit_gmm_marker_via_subgenes,
    plot_gmm_marker_subgene_analysis,
    run_inhibitory_gmm_thresholding,
    filter_cells_by_gmm_thresholds,
    plot_cluster_centroids
)
from .cells import (
    plot_single_cell_expression_all_rounds,
)
from .segmentation import (
    fig_cell_centroids_comparison,
    fig_centroids_filtered,
    plot_centroids,
    plot_single_cell_segmentation_overview,
    qc_segmentation,
)
from .spectral_unmixing import (
    plot_pairwise_intensities_multi_ratios,
    plot_filtered_intensities,
    plot_dye_lines_pairwise,
    plot_channel_intensity_histograms,
    plot_channel_intensity_histograms_by_round,
)
from .tile_alignment import (
    qc_tile_alignment,
)
from .cluster_similarity import (
    compute_cluster_similarity,
    plot_cluster_similarity,
    plot_similarity_heatmap,
    plot_match_summary,
    plot_matched_centroid_heatmap,
)
from .single_cell_unmixing import (
    plot_spot_projection,
    plot_spot_measure_distributions,
    plot_cell_qc,
    plot_spot_nn_distances,
    plot_adjacent_channel_scatter,
    select_focused_exemplars,
    annotate_spots_with_valid,
    build_round_cell_summary,
)

__all__ = [
    "CHANNEL_COLORS",
    # camera alignment
    "qc_camera_alignment",
    # tile alignment
    "qc_tile_alignment",
    # cell x gene
    "plot_cell_x_gene_simple",
    "plot_cell_x_gene_clustered",
    "fig_mixed_unmixed_cxg_and_corr",
    "plot_cluster_centroids"
    # gmm thresholding
    "compute_gmm_bic_table",
    "plot_bic_comparison",
    "fit_gmm_threshold",
    "threshold_genes",
    "plot_gmm_threshold_grid",
    "fit_gmm_marker_via_subgenes",
    "plot_gmm_marker_subgene_analysis",
    "run_inhibitory_gmm_thresholding",
    "filter_cells_by_gmm_thresholds",
    # segmentation
    "qc_segmentation",
    "plot_single_cell_segmentation_overview",
    "plot_centroids",
    "fig_cell_centroids_comparison",
    "fig_centroids_filtered",
    # spectral unmixing
    "plot_pairwise_intensities_multi_ratios",
    "plot_filtered_intensities",
    "plot_dye_lines_pairwise",
    "plot_channel_intensity_histograms",
    "plot_channel_intensity_histograms_by_round",
    # cells
    "plot_single_cell_expression_all_rounds",
    # cluster similarity
    "compute_cluster_similarity",
    "plot_cluster_similarity",
    "plot_similarity_heatmap",
    "plot_match_summary",
    "plot_matched_centroid_heatmap",
    # single-cell unmixing QC
    "plot_spot_projection",
    "plot_spot_measure_distributions",
    "plot_cell_qc",
    "plot_spot_nn_distances",
    "plot_adjacent_channel_scatter",
    "select_focused_exemplars",
    "annotate_spots_with_valid",
    "build_round_cell_summary",
]
