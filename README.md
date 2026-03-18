# hcr_data_qc

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-86.4%25-yellow)
![Coverage](https://img.shields.io/badge/coverage-7%25-red?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

## About

Quality control analysis for AIND HCR data processing. Provides tools for validating tile alignment, camera alignment, segmentation, spectral unmixing, and spot detection.

**Use Cases:**
1. **Interactive analysis** - Within a CodeOcean capsule/cloud workstation (or local machine):

    + Make sure processed HCR data asset is attached to capsule


   ```bash
   // run just tile_alignment
   python launch_qc.py --dataset HCR_788639-25_2025-06-06_13-00-00_processed_2025-06-17_07-08-14 --output-dir /root/capsule/scratch/qc-test --tile-alignment --pyramid-level 4
   ```

   ```bash
   // run all qc 
   python launch_qc.py --dataset HCR_788639-25_2025-06-06_13-00-00_processed_2025-06-17_07-08-14 --output-dir /root/capsule/scratch/qc-test --all --pyramid-level 0 
   ```
2. **Reproducible runs** - With CodeOcean app panel
    + See [HCR QC Kickoff capsule](https://codeocean.allenneuraldynamics.org/capsule/8714887/tree)
4. **Pipeline integration** - As automated QC steps *(not implemented yet)*

Intergration will AIND QC portal will happen when team identifies and evaluates essential plots.

## Change log

**v0.5.0 (03/18/2026)**

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
+  new metadata parsing and accessors in HCRRound/HCRDataset
+ improved loading spot functions, cleaned up returned dataframe
+ added get_spot_channel_gene_map()
+ added simple soma classifier (logistic regression)
+ added HCR cell filters, to remove non-somas (classifier of choice) and overlap cells

**v0.3.9 (8/11/2025)**
+ Add spot detection visualization function to plot top 10 cells per gene based on spot count
+ Implement segmentation overview plotting with single and multi-view options
+ Create comprehensive cell expression plotting functions for multi-round HCR analysis
+ Extract and modularize zarr data processing utilities for cell data and masks
+ Example notebook for single cell plots

**v0.3.8 (8/08/2025)**
+ refactor file tree to use "viz" api for simplified access to plotting functions

**v0.3.7 (8/01/2025)**
+ Adds a new plot_spot_metric_dist function for analyzing spot quality distributions with correlation and distance thresholds
+ Introduces cluster-based centroid plotting capabilities and cell x gene analysis for coregistered spots
+ Updates channel color mapping and improves plot layout consistency

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

## Contributing

### Linters and testing
Run `pre_commit_checks.py`, which includes coverage, black, isort, flake8, & interrogate

### Pull requests

+ **Internal members** please create a branch. 
+ **External members** please fork the repository and open a pull request from the fork. 

### Commits
We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>: <short summary>
```

type is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests
