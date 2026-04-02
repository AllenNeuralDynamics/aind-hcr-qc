# Changelog

**v0.7.0 (04/02/2026)**

*New module: `utils/s3_qc.py`*
+ S3 CRUD helpers for QC plot management: `check_plot_exists()`, `upload_plot()`, `download_plot()`, `delete_plot()`, `list_plots()`, `load_plot_metadata()`
+ Canonical path: `s3://aind-scratch-data/ctl/hcr/qc/{mouse_id}/{plot_type}.png` with JSON sidecar

*New module: `viz/intergrated_datasets.py`*
+ `add_unmixed_channel_intensity()` — maps each spot to its own-channel intensity with optional threshold filter
+ `plot_intensity_violins()` — per-channel/gene intensity violin plots for integrated spot data
+ `plot_gene_spot_count_pairplot()` — pairplot of gene spot counts across rounds

*`viz/cell_x_gene.py`*
+ Added `invert_z` parameter and `_scatter_params_for_n()` adaptive scatter helper to `plot_cluster_centroids()`
+ Added `plot_all_subclass_centroids()` for batch subclass centroid visualization
+ Fixed `groupby` aggregation for pandas 3 compatibility in violin plots

*Tests & docs*
+ `tests/test_s3_qc.py` — unit tests for S3 QC utilities
+ `docs/panel_qc_viewer_design.md` — design doc for the panel QC viewer

**v0.6.0 (03/18/2026)**

*New module: `viz/cluster_similarity.py`*
+ `compute_cluster_similarity()` — pairwise cluster comparison between rounds using expression centroids and cosine/correlation similarity
+ `match_clusters()` — greedy best-match assignment between two sets of clusters
+ `plot_similarity_heatmap()`, `plot_match_summary()`, `plot_matched_centroid_heatmap()` — visualizations for cluster correspondence
+ `run_pairwise_comparisons()` / `summarise_pairwise_comparisons()` — batch all-vs-all round comparisons with summary plots
+ All cluster-similarity functions exported from `viz.__init__`

*`viz/cell_x_gene.py`*
+ Added full GMM thresholding pipeline for inhibitory marker genes
+ Added `fig_mixed_unmixed_cxg_and_corr()` — side-by-side mixed/unmixed CxG with correlation panel
+ Added `cluster_cells()` — k-means / GMM clustering helper with BIC-based optimal-k selection (guarded against small datasets)
+ `plot_cell_x_gene_simple()` updated: gene-label building refactored into `_build_gene_labels()`, `round_channel_gene` label support, cluster-label overlays
+ All new CxG functions exported from `viz.__init__`

*`viz/spectral_unmixing.py`*
+ Added spot-reassignment analysis: `create_reassignment_matrix()`, `plot_reassignment_matrix()`, `analyze_spot_fate()`, `plot_spot_fate()`, `fig_unmixing_comprehensive()`
+ Added cross-channel nearest-neighbour stats: `cross_channel_nn()`, `cross_channel_nn_gpu()` (optional torch backend), `compute_channel_pair_proximity_stats()`, `compute_all_channel_pair_proximity_stats()`, `compute_all_rounds_proximity_stats()`
+ Added proximity visualization: `plot_proximity_stats_heatmap()`, `plot_proximity_stats_comparison()`, `plot_all_channel_nn_histograms()`
+ Added `create_unique_spot_id()` utility
+ `torch` import made optional inside `cross_channel_nn_gpu` (graceful fallback)

*`viz/segmentation.py`*
+ `plot_centroids()` gains `vmin`/`vmax` parameters for explicit color-scale control
+ Fixed y-axis inversion: no longer inverts for the `XY` plane (only `ZX`/`ZY`)

*`io/zarr_data.py`*
+ Added `extract_volume_around_point()` and `extract_plane_around_point()` for extracting zarr sub-volumes centered on a coordinate

*`utils/utils.py`*
+ Added `combine_pngs_to_pdf()` — natural-sorted PNG-to-PDF combiner for batch figure export
+ Added `_natural_sort_key()` helper

*`scripts/batch_unmixing_analysis.py`* (new)
+ Batch script to run comprehensive unmixing analysis figures across multiple mice and save output as PDFs

**v0.4.0 (10/16/2025)**
+ New metadata parsing and accessors in HCRRound/HCRDataset
+ Improved loading spot functions, cleaned up returned dataframe
+ Added `get_spot_channel_gene_map()`
+ Added simple soma classifier (logistic regression)
+ Added HCR cell filters to remove non-somas and overlap cells

**v0.3.9 (8/11/2025)**
+ Add spot detection visualization function to plot top 10 cells per gene based on spot count
+ Implement segmentation overview plotting with single and multi-view options
+ Create comprehensive cell expression plotting functions for multi-round HCR analysis
+ Extract and modularize zarr data processing utilities for cell data and masks
+ Example notebook for single cell plots

**v0.3.8 (8/08/2025)**
+ Refactor file tree to use "viz" api for simplified access to plotting functions

**v0.3.7 (8/01/2025)**
+ Adds a new `plot_spot_metric_dist` function for analyzing spot quality distributions with correlation and distance thresholds
+ Introduces cluster-based centroid plotting capabilities and cell x gene analysis for coregistered spots
+ Updates channel color mapping and improves plot layout consistency
