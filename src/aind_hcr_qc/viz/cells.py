# -*- coding: utf-8 -*-
"""
Plotting functions for visualizing single cell expression data across multiple HCR rounds.
"""
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from aind_hcr_data_loader.hcr_dataset import HCRDataset
from aind_hcr_qc.utils.utils import saveable_plot

import aind_hcr_qc.io.zarr_data as zarr_data

# -------------------------------------------------------------------------------------------------
# Cache for expensive operations
# -------------------------------------------------------------------------------------------------

_channel_gene_cache: Dict[int, dict] = {}


def _get_cached_channel_gene_table(dataset: HCRDataset) -> dict:
    """Cache the channel-gene mapping to avoid repeated CSV loading."""
    dataset_id = id(dataset)
    if (dataset_id not in _channel_gene_cache) or (_channel_gene_cache[dataset_id] is None):
        try:
            # Use spots_only=False to include cellular markers like Rn28s (405)
            channel_gene_table = dataset.create_channel_gene_table(spots_only=False)
            _channel_gene_cache[dataset_id] = channel_gene_table
        except Exception:
            _channel_gene_cache[dataset_id] = None
    return _channel_gene_cache[dataset_id]


def _load_channel_crop(args) -> Tuple[str, Optional[np.ndarray]]:
    """Helper function for parallel channel loading."""
    chan, dataset, round_key, pyramid_level, origin, crop_shape = args
    try:
        chan_zarr = dataset.load_zarr_channel(round_key, chan, data_type="fused", pyramid_level=pyramid_level)
        z0, y0, x0 = origin
        z1, y1, x1 = z0 + crop_shape[0], y0 + crop_shape[1], x0 + crop_shape[2]
        chan_crop = np.asarray(chan_zarr[0, 0, z0:z1, y0:y1, x0:x1])
        return chan, chan_crop
    except Exception:
        return chan, None

# -------------------------------------------------------------------------------------------------
# Multi Round
# -------------------------------------------------------------------------------------------------


def plot_cell_gene_expression_ax(cell_id, cellxgene_df, gene_order=None, ax=None, unique_roi_id=None):
    """
    Plot a barplot of gene expression for a single cell.
    
    Parameters
    ----------
    cell_id : int
        The cell ID to plot
    cellxgene_df : pd.DataFrame
        DataFrame with cell_id as index and gene columns with spot counts
    gene_order : list, optional
        Order of genes to plot. If None, uses all columns except cluster_id/cluster_label
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure
    unique_roi_id : int or str, optional
        Unique ROI ID from coreg table to display in the title
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes with the barplot
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get cell data
    if cell_id not in cellxgene_df.index:
        ax.text(0.5, 0.5, f"Cell {cell_id} not found", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return ax
    
    cell_data = cellxgene_df.loc[cell_id]
    
    # Determine gene columns (exclude metadata columns)
    exclude_cols = ['cluster_id', 'cluster_label', 'supertype_name']
    if gene_order is None:
        gene_order = [col for col in cellxgene_df.columns if col not in exclude_cols]
    
    # Get values for each gene
    values = [cell_data.get(gene, 0) for gene in gene_order]
    
    # Create barplot
    bars = ax.bar(range(len(gene_order)), values, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(gene_order)))
    ax.set_xticklabels(gene_order, rotation=90, ha='center', fontsize=18)
    ax.set_ylabel('Spot count', fontsize=22)
    ax.set_xlabel('')
    
    # Add cluster label if available
    if 'cluster_label' in cellxgene_df.columns:
        cluster_label = cell_data.get('cluster_label', '')
        if unique_roi_id is not None:
            ax.set_title(f"HCR ID: {cell_id} - {cluster_label} - ROIcat ID: {unique_roi_id}", fontsize=22)
        else:
            ax.set_title(f"HCR ID: {cell_id} - {cluster_label}", fontsize=22)
    else:
        if unique_roi_id is not None:
            ax.set_title(f"HCR ID: {cell_id} - Unique ROIcat ID: {unique_roi_id}", fontsize=22)
        else:
            ax.set_title(f"HCR ID: {cell_id}", fontsize=22)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust the axes position to leave room for x-tick labels within the allocated space
    # This prevents the labels from extending beyond the subplot area
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', labelsize=16)
    
    return ax


@saveable_plot()
def plot_single_cell_expression_all_rounds(
    plot_cell_id: int, dataset: HCRDataset, 
    pyramid_level: str = "0", 
    rounds: List[str] = None, 
    vmin_vmax = "auto", 
    verbose: bool = False,
    linear_unmix_matrix=None,
    cellxgene_df=None,
    gene_order=None,
    coreg_table=None,
) -> plt.Figure:
    """
    Plot single cell expression across multiple HCR rounds in a compact vertical layout.

    Creates a multi-panel figure showing all channels for a specific cell across different
    imaging rounds. Each round is displayed as a horizontal row of channel images with
    minimal spacing between rounds for easy comparison.

    Parameters
    ----------
    plot_cell_id : int
        The cell ID to visualize across rounds
    dataset : HCRDataset
        The HCR dataset object containing the imaging data
    rounds : list of str
        List of round identifiers (e.g., ["R1", "R2", "R3"])
    pyramid_level : str, optional
        Zarr pyramid level for image resolution, by default "0" (full resolution)
    verbose : bool, optional
        Whether to print detailed processing information, by default False
    linear_unmix_matrix : np.ndarray, optional
        Matrix for linear unmixing of channels
    cellxgene_df : pd.DataFrame, optional
        DataFrame with cell_id as index and gene expression columns. If provided,
        the first row will include a gene expression barplot taking up 2 panels.
    gene_order : list, optional
        Order of genes for the barplot. If None, uses all gene columns.

    Returns
    -------
    plt.Figure
        The combined figure containing all rounds as subfigures

    Notes
    -----
    - Uses automatic intensity scaling (5th-95th percentile) for optimal visualization
    - Displays segmentation mask overlays with transparency
    - Trims images to square aspect ratio for consistent appearance
    - Uses tight layout with minimal spacing between rounds
    - Each round is displayed as a subfigure with channel titles showing gene names
    - Handles missing data gracefully with error messages
    - If cellxgene_df is provided, the first row (R1) will show a barplot in the first 2 panels

    Examples
    --------
    >>> fig = plot_single_cell_expression_all_rounds(
    ...     plot_cell_id=12345,
    ...     rounds=["R1", "R2", "R3", "R4"],
    ...     pyramid_level="0",
    ...     dataset=my_hcr_dataset,
    ...     verbose=True
    ... )
    >>> plt.show()

    See Also
    --------
    gene_plotter.plot_all_channels_cell : Individual round plotting function
    """
    if rounds is None:
        rounds = dataset.rounds.keys() if dataset else []
    if not isinstance(rounds, list):
        raise ValueError("Rounds must be a list of round identifiers (e.g., ['R1', 'R2', ...])")

    # Determine if we're adding the barplot to the first row
    include_barplot = cellxgene_df is not None

    # Create a single parent figure
    fig = plt.figure(figsize=(20, 4 * len(rounds)))

    # Create a GridSpec layout
    gs = gridspec.GridSpec(
        len(rounds), 1, figure=fig, hspace=0.05, wspace=0, top=0.95, bottom=0.05, left=0.05, right=0.95
    )

    # Create subfigures for each round
    for i, round_n in enumerate(rounds):
        # Create a subfigure from the gridspec
        subfig = fig.add_subfigure(gs[i, :])

        # For the first round with barplot, create a custom layout
        if i == 0 and include_barplot:
            try:
                # Get channel count to determine layout
                channels = dataset.get_channels(round_n)
                
                # Use channel_gene_table to determine which channels have genes
                # This handles cases like R1 where some channels don't have genes (e.g., Syto59)
                channel_gene_table = _get_cached_channel_gene_table(dataset)
                if channel_gene_table is not None:
                    # Get channels that have genes for this round (from spots_only=False table)
                    # Filter to only include channels with actual gene probes
                    round_col = "Round" if "Round" in channel_gene_table.columns else "round"
                    channel_col = "Channel" if "Channel" in channel_gene_table.columns else "channel"
                    gene_col = "Gene" if "Gene" in channel_gene_table.columns else "gene"
                    
                    round_genes = channel_gene_table[channel_gene_table[round_col] == round_n]
                    # Get channels that have genes (exclude non-gene markers like Syto59)
                    # Filter out known non-gene markers
                    exclude_markers = ["Syto59", "Syto", "SYTO"]
                    round_genes_filtered = round_genes[~round_genes[gene_col].str.contains('|'.join(exclude_markers), case=False, na=False)]
                    gene_channels = [str(ch) for ch in round_genes_filtered[channel_col].values]
                    # For R1, also include 405 channel (cellular marker) at the beginning
                    if round_n == "R1" and "405" in channels and "405" not in gene_channels:
                        gene_channels = ["405"] + gene_channels
                    # Sort by channel number to maintain consistent order
                    gene_channels = sorted(gene_channels, key=lambda x: int(x))
                    if verbose:
                        print(f"Gene channels for {round_n}: {gene_channels}")
                else:
                    # Fallback: use all channels except known non-gene channels
                    gene_channels = sorted(channels, key=lambda x: int(x))
                
                n_gene_channels = len(gene_channels)
                
                # Create GridSpec within subfigure with 2 rows:
                # Row 0 (top half): image panels and barplot data area
                # Row 1 (bottom half): empty space for barplot x-axis labels
                # First n_gene_channels columns for image panels, 1 spacer column, last 3 columns for barplot
                total_cols = n_gene_channels + 1 + 3  # images + spacer + barplot
                # Use width_ratios to make the spacer column narrower
                width_ratios = [1] * n_gene_channels + [0.3] + [1, 1, 1]  # spacer is 0.3 width
                # Use height_ratios: top row gets 50%, bottom row for labels gets 50%
                inner_gs = gridspec.GridSpec(2, total_cols, figure=subfig, wspace=0.05, hspace=0.0, 
                                            width_ratios=width_ratios, height_ratios=[1, 1])
                
                # Extract round number from round_n (e.g., "R1" -> "1")
                round_num = round_n.replace("R", "") if round_n.startswith("R") else round_n
                
                # Plot gene channels in columns 0 to (n_gene_channels - 1)
                plot_all_channels_cell(
                    dataset=dataset,
                    round_key=round_n,
                    cell_id=plot_cell_id,
                    pyramid_level=pyramid_level,
                    vmin_vmax=vmin_vmax,
                    plot_mask_outlines=True,
                    trim_to_square=True,
                    figsize=None,
                    verbose=verbose,
                    fig=subfig,
                    linear_unmix_matrix=linear_unmix_matrix,
                    row_label=f"Round {round_num}",
                    start_col_index=0,  # Start at column 0
                    total_cols=total_cols,  # Total number of columns in the grid (includes spacer)
                    max_channels=n_gene_channels,  # Plot all gene channels
                    include_channels=gene_channels,  # Only include these specific channels
                    inner_gs=inner_gs,  # Pass the GridSpec with custom width_ratios
                )
                
                # Create axes for barplot (spans last 3 columns, after the spacer)
                # Use only row 0 (top half) for the barplot
                ax_barplot = subfig.add_subplot(inner_gs[0, -3:])
                
                # Get unique_roi_id from coreg_table if provided
                unique_roi_id = None
                if coreg_table is not None:
                    try:
                        unique_roi_id = coreg_table[coreg_table.hcr_id == plot_cell_id].unique_roicat_id.values[0]
                    except (KeyError, IndexError):
                        pass
                
                plot_cell_gene_expression_ax(plot_cell_id, cellxgene_df, gene_order=gene_order, ax=ax_barplot, unique_roi_id=unique_roi_id)
                
                # Adjust barplot position to align with image panels and leave room for x-labels
                # Get the position and shrink it to leave room at the bottom for labels
                pos = ax_barplot.get_position()
                # Shrink height by 50% from the bottom to leave room for rotated x-tick labels
                new_height = pos.height * 0.50
                new_bottom = pos.y0 + (pos.height - new_height)  # Move up by the amount we shrunk
                ax_barplot.set_position([pos.x0, new_bottom, pos.width, new_height])
            except Exception as e:
                print(f"Error plotting round {round_n} for cell {plot_cell_id}: {e}")
                import traceback
                traceback.print_exc()
                subfig.add_subplot(111).text(0.5, 0.5, f"Error: {e}", fontsize=12, ha="center", va="center")
                plt.axis("off")
                continue
        else:
            try:
                # Extract round number from round_n (e.g., "R1" -> "1")
                round_num = round_n.replace("R", "") if round_n.startswith("R") else round_n
                
                # Plot directly on the subfigure
                plot_all_channels_cell(
                    dataset=dataset,
                    round_key=round_n,
                    cell_id=plot_cell_id,
                    pyramid_level=pyramid_level,
                    vmin_vmax=vmin_vmax,
                    plot_mask_outlines=True,
                    trim_to_square=True,
                    figsize=None,
                    verbose=verbose,
                    fig=subfig,
                    linear_unmix_matrix=linear_unmix_matrix,
                    row_label=f"Round {round_num}"
                )
            except Exception as e:
                print(f"Error plotting round {round_n} for cell {plot_cell_id}: {e}")
                subfig.add_subplot(111).text(0.5, 0.5, f"Error: {e}", fontsize=12, ha="center", va="center")
                plt.axis("off")
                plt.tight_layout()
                continue

    # Determine if using fixed LUT
    is_fixed_lut = isinstance(vmin_vmax, (tuple, list)) and len(vmin_vmax) == 2
    
    # Add overall title - include LUT if fixed
    if is_fixed_lut:
        fig.suptitle(f"Cell ID {plot_cell_id}  |  LUT=[{vmin_vmax[0]:.0f}-{vmin_vmax[1]:.0f}]", fontsize=18, y=1.05)
    else:
        fig.suptitle(f"Cell ID {plot_cell_id}", fontsize=18, y=1.05)

    plt.tight_layout()
    return fig


def linear_unmix(
    image,
    mix_matrix,
    axis=-1,
    method="auto",      # "auto" | "inv" | "pinv" | "nnls"
    rcond=1e-6,         # used for pinv
    offsets=None,       # per-observed-channel baseline to subtract
    clip=True,          # clip negatives to 0 for inv/pinv paths
    out_dtype=None,     # e.g., np.float32
):
    """
    Unmix multi-channel fluorescence images given a crosstalk (mixing) matrix.

    Model (per pixel): observed = true @ M
      - image[..., c_obs]
      - true has C_true channels; usually C_true == C_obs == C
      - If your lab defines observed = M @ true instead, pass M.T

    Parameters
    ----------
    image : np.ndarray
        Shape (..., C) or (C, ...).
    mix_matrix : array-like (C_true, C_obs)
        M in observed = true @ M.
    axis : int
        Channel axis in `image`.
    method : {"auto","inv","pinv","nnls"}
        * "auto": inv if square & well-conditioned else pinv
        * "inv":  matrix inverse (square only)
        * "pinv": Moore–Penrose pseudo-inverse (least squares)
        * "nnls": Non-Negative Least Squares per pixel (requires SciPy)
    rcond : float
        Cutoff for small singular values in pinv.
    offsets : array-like or None
        Per-observed-channel baseline to subtract before unmixing (length = C_obs).
    clip : bool
        Clip negatives to 0 for inv/pinv paths (ignored for nnls, which returns ≥0).
    out_dtype : np.dtype or None
        Output dtype (default float32).

    Returns
    -------
    unmixed : np.ndarray
        Same shape as `image`, unmixed channels on `axis`.
    """
    img = np.asarray(image)
    M = np.asarray(mix_matrix, dtype=np.float64)
    img_ch_last = np.moveaxis(img, axis, -1).astype(np.float64, copy=False)

    # Sanity checks
    C_obs = img_ch_last.shape[-1]
    if M.shape[1] != C_obs:
        raise ValueError(f"mix_matrix second dimension ({M.shape[1]}) must equal observed channels ({C_obs}).")

    # Optional baseline subtraction on observed channels
    if offsets is not None:
        offsets = np.asarray(offsets, dtype=np.float64)
        if offsets.shape[0] != C_obs:
            raise ValueError("`offsets` must have length equal to observed channels.")
        img_ch_last = img_ch_last - offsets

    # Prepare observed as (N_pixels, C_obs)
    leading_shape = img_ch_last.shape[:-1]
    O = img_ch_last.reshape(-1, C_obs)

    if method.lower() == "nnls":
        # Solve: min || A x - b ||^2 s.t. x>=0, where A = M^T, x = true^T, b = observed^T
        try:
            from scipy.optimize import nnls
        except Exception as e:
            raise ImportError(
                "method='nnls' requires SciPy (scipy.optimize.nnls). "
                "Install scipy or use method='pinv'/'auto'."
            ) from e

        A = M.T  # shape (C_obs, C_true)
        C_true = M.shape[0]
        T = np.empty((O.shape[0], C_true), dtype=np.float64)
        # Simple per-pixel loop; for speed on huge images, consider batching/parallelism
        for i in range(O.shape[0]):
            T[i], _ = nnls(A, O[i])
    else:
        # Compute unmixing matrix U such that true = observed @ U
        # We have observed (N x C_obs), want true (N x C_true): U = M^{-1} (if square) or M^{+}
        use_inv = (method in ("auto", "inv")) and (M.shape[0] == M.shape[1])
        if use_inv:
            try:
                cond = np.linalg.cond(M)
                if method == "inv" or cond < 1 / rcond:
                    U = np.linalg.inv(M)
                else:
                    U = np.linalg.pinv(M, rcond=rcond)
            except np.linalg.LinAlgError:
                U = np.linalg.pinv(M, rcond=rcond)
        elif method == "pinv" or method == "auto":
            U = np.linalg.pinv(M, rcond=rcond)
        else:
            raise ValueError("Unknown method. Use 'auto', 'inv', 'pinv', or 'nnls'.")

        T = O @ U
        if clip:
            np.maximum(T, 0, out=T)

    # Reshape back to image layout
    if out_dtype is None:
        out_dtype = np.float32
    T = T.reshape(*leading_shape, M.shape[0]).astype(out_dtype, copy=False)
    return np.moveaxis(T, -1, axis)


# -------------------------------------------------------------------------------------------------
# High level drivers
# -------------------------------------------------------------------------------------------------
def plot_all_channels_cell(
    dataset,
    round_key,
    cell_id,
    pyramid_level="0",
    vmin_vmax="auto",
    plot_mask_outlines=True,
    num_planes=5,
    plot_buffer=50,
    figsize=None,
    trim_to_square=True,
    fig=None,
    verbose=False,
    linear_unmix_matrix=None,
    row_label=None,
    start_col_index=0,
    total_cols=None,
    max_channels=None,
    skip_channels=None,
    include_channels=None,
    inner_gs=None,
):
    """
    Plot segmentation crops for all channels in an HCRDataset round for a specific cell.

    Parameters:
    -----------
    dataset : HCRDataset
        The HCR dataset object
    round_key : str
        Round identifier (e.g., 'R1')
    cell_id : int
        Cell ID to plot
    pyramid_level : str
        Pyramid level for zarr data
    vmin_vmax : str or tuple
        Either "auto" for 5th-95th percentile, or tuple (vmin, vmax) for fixed range,
        or 'blend' which is fixed for all channels, unless its max is above the 99& for that channel, then use that channel's 99%
    plot_mask_outlines : bool
        Whether to overlay segmentation mask outlines
    num_planes : int
        Number of z-planes to extract for plane selection
    plot_buffer : int
        Buffer around cell for cropping
    figsize : tuple, optional
        Figure size. If None, calculated automatically
    trim_to_square : bool
        Whether to trim images to square aspect ratio (default: True)
    fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure, optional
        Existing figure or subfigure to plot into. If None, creates a new figure.
    row_label : str, optional
        Label to display as ylabel on the first axis (e.g., "Round R1")
    start_col_index : int, optional
        Starting column index for placing axes (default: 0). Used when other elements
        occupy the first columns of the grid.
    total_cols : int, optional
        Total number of columns in the grid. If None, uses number of channels.
    max_channels : int, optional
        Maximum number of channels to plot. If None, plots all channels.
    skip_channels : list, optional
        List of channel names to skip (e.g., ["405"] to skip reference channel).
    include_channels : list, optional
        List of channel names to include. If provided, only these channels will be plotted
        (takes precedence over skip_channels).
    inner_gs : matplotlib.gridspec.GridSpec, optional
        Pre-created GridSpec to use for placing axes. If provided, axes will be created
        using this GridSpec instead of creating new subplots.

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure (or the input fig if provided)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(cell_id, str):
        try:
            cell_id = int(cell_id)
        except ValueError:
            raise ValueError(f"Invalid cell_id: {cell_id}. Must be an integer.")
    if verbose:
        print(f"\n{'='*60}")
        print(f"PLOTTING ALL CHANNELS FOR CELL {cell_id} - ROUND {round_key}")
        print(f"{'='*60}")

    # Get available channels and sort them
    channels = dataset.get_channels(round_key)
    channels_sorted = sorted(channels, key=lambda x: int(x))
    if verbose:
        print(f"Available channels: {channels_sorted}")

    # Get gene mapping for channel titles
    channel_gene_table = _get_cached_channel_gene_table(dataset)
    if channel_gene_table is not None:
        # Handle both capitalized and lowercase column names
        round_col = "Round" if "Round" in channel_gene_table.columns else "round"
        channel_col = "Channel" if "Channel" in channel_gene_table.columns else "channel"
        gene_col = "Gene" if "Gene" in channel_gene_table.columns else "gene"
        
        round_genes = channel_gene_table[channel_gene_table[round_col] == round_key]
        channel_to_gene = dict(zip(round_genes[channel_col].astype(str), round_genes[gene_col]))
        if verbose:
            print(f"Channel-gene mapping: {channel_to_gene}")
    else:
        if verbose:
            print(f"Warning: Could not load gene mapping")
        channel_to_gene = {}

    # Load cell info and segmentation data
    if verbose:
        print("\nLoading cell info and segmentation data...")
    # cell_info_df = dataset.get_cell_info(source="segmentation")
    cell_info_df = dataset.rounds[round_key].get_cell_info(source="mixed_cxg")
    if verbose:
        print(cell_info_df.describe())
    cell_info_array = cell_info_df[["z_centroid", "y_centroid", "x_centroid", "cell_id"]].to_numpy()
    segmentation_zarr = dataset.load_segmentation_mask(round_key, pyramid_level)

    # Get reference channel for segmentation overlay (usually 405)
    ref_channel = "405" if "405" in channels else channels_sorted[0]
    if verbose:
        print(f"Using reference channel {ref_channel} for segmentation overlay")

    # Extract cell volume using reference channel
    if verbose:
        print(f"Extracting cell volume for cell {cell_id}...")
    ref_zarr = dataset.load_zarr_channel(round_key, ref_channel, data_type="fused", pyramid_level=pyramid_level)
    if verbose:
        print(f"Reference channel shape: {ref_zarr.shape}")

    seg_crop, img_crop, masks_only, cell_mask_only, origin, z_planes, x_planes = zarr_data.extract_cell_volume(
        segmentation_zarr, ref_zarr, cell_info_array, cell_id, num_planes=num_planes, plot_buffer=plot_buffer, verbose=verbose
    )

    cell_centroid = cell_info_array[cell_info_array[:, -1] == cell_id, :-1][0]
    if verbose:
        print(f"Cell centroid (z, y, x): {cell_centroid}")

        print(f"Cell crop shape: {seg_crop.shape}")
        print(f"Origin: {origin}")
        print(f"Z-planes: {z_planes}")

    # Load all channel data
    if verbose:
        print("\nLoading channel data...")
    channel_arrays = {}
    with ThreadPoolExecutor() as executor:
        args = [(chan, dataset, round_key, pyramid_level, origin, seg_crop.shape) for chan in channels_sorted]
        results = executor.map(_load_channel_crop, args)
        for chan, chan_crop in results:
            if chan_crop is not None:
                channel_arrays[chan] = chan_crop
                if verbose:
                    print(f"  Channel {chan}: loaded, shape {chan_crop.shape}")
            else:
                if verbose:
                    print(f"  Channel {chan}: failed to load")
    if verbose:
        print(f"Successfully loaded {len(channel_arrays)} channels")

    if (linear_unmix_matrix is not None) and (round_key != 'R1'):
        chan_names = list(channel_arrays.keys())
        img_arrays = list(channel_arrays.values())
        # if channel =405, remove
        if "405" in chan_names:
            idx_405 = chan_names.index("405")
            chan_names.pop(idx_405)
            img_arrays.pop(idx_405)
            # print("Removing channel 405 for linear unmixing")
        img_stack = np.stack(img_arrays, axis=-1)  # shape (Y, X, C)
        if verbose:
            print(f"Applying linear unmixing with matrix shape {linear_unmix_matrix.shape}...")
        unmixed_stack = linear_unmix(
            img_stack,
            linear_unmix_matrix,
            axis=-1,
            method="auto",
            rcond=1e-6,
            offsets=None,
            clip=True,
            out_dtype=np.float32,
        )

        for i, chan in enumerate(chan_names):
            channel_arrays[chan] = unmixed_stack[..., i]
            if verbose:
                print(f"  Channel {chan}: unmixed")
        if verbose:
            print("Linear unmixing completed.")
    # -------------------------------------------------------------------------------------------------


    # Calculate figure layout - single row
    n_channels = len(channel_arrays)
    if n_channels == 0:
        print("No channels loaded successfully!")
        return None

    cols = total_cols if total_cols is not None else n_channels  # Use total_cols if provided
    rows = 1

    if figsize is None:
        figsize = (cols * 4, 5)  # Fixed height for single row

    if verbose:
        print(f"\nCreating figure: {rows}x{cols} grid, figsize={figsize}")

    # Create figure or use provided one
    if fig is None:
        fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
        if n_channels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()  # Ensure it's always a 1D array
    else:
        # Use provided figure/subfigure - create subplots within it
        # Determine how many axes to create based on max_channels or remaining space
        if max_channels is not None:
            n_axes_to_create = max_channels
        else:
            n_axes_to_create = cols - start_col_index
        axes = []
        for i in range(n_axes_to_create):
            if inner_gs is not None:
                # Use the provided GridSpec - span all rows if multi-row
                ax = fig.add_subplot(inner_gs[:, start_col_index + i])
            else:
                ax = fig.add_subplot(rows, cols, start_col_index + i + 1)
            axes.append(ax)

    # Select middle z-plane for display
    middle_z = z_planes[len(z_planes) // 2]
    if verbose:
        print(f"Plotting middle z-plane: {middle_z} (global z: {origin[0] + middle_z})")

    # Determine if using fixed LUT (same for all channels)
    is_fixed_lut = isinstance(vmin_vmax, (tuple, list)) and len(vmin_vmax) == 2
    if is_fixed_lut:
        fixed_vmin, fixed_vmax = vmin_vmax

    # Plot each channel
    # Determine which channels to plot based on include_channels, skip_channels, etc.
    channels_to_plot = list(channel_arrays.keys())
    
    # If include_channels is specified, only include those channels (takes precedence)
    if include_channels is not None:
        channels_to_plot = [ch for ch in channels_to_plot if ch in include_channels]
        # Sort by the order in include_channels to maintain specified order
        channels_to_plot = sorted(channels_to_plot, key=lambda x: include_channels.index(x) if x in include_channels else float('inf'))
    elif skip_channels is not None:
        # Skip specified channels (e.g., 405 reference channel)
        channels_to_plot = [ch for ch in channels_to_plot if ch not in skip_channels]
    
    if start_col_index > 0 and total_cols is not None:
        # Only plot channels that fit in the remaining space
        n_channels_to_plot = total_cols - start_col_index
        channels_to_plot = channels_to_plot[:n_channels_to_plot]
    
    if max_channels is not None:
        channels_to_plot = channels_to_plot[:max_channels]

    for i, chan in enumerate(channels_to_plot):
        ax = axes[i]

        chan_data = channel_arrays[chan]

        # Calculate vmin/vmax
        if vmin_vmax == "auto":
            vmin = np.percentile(chan_data, 5)
            vmax = np.percentile(chan_data, 99.9)
        elif vmin_vmax == "blend":
            # fixed at 90
            vmin = 90
            vmax_99 = np.percentile(chan_data, 99.95)
            if vmax_99 > 600:
                vmax = vmax_99
            else:
                vmax = 600
        elif is_fixed_lut:
            vmin, vmax = fixed_vmin, fixed_vmax
        else:
            vmin, vmax = chan_data.min(), chan_data.max()

        if verbose:
            print(f"  Channel {chan}: vmin={vmin:.1f}, vmax={vmax:.1f}")

        # Get the middle z-plane
        img_slice = chan_data[middle_z, :, :]

        # Trim to square if requested
        if trim_to_square:
            h, w = img_slice.shape
            min_dim = min(h, w)

            # Calculate center crop coordinates
            y_start = (h - min_dim) // 2
            x_start = (w - min_dim) // 2
            y_end = y_start + min_dim
            x_end = x_start + min_dim

            # Crop the image and masks
            img_slice = img_slice[y_start:y_end, x_start:x_end]

            if verbose:
                print(f"  Channel {chan}: trimmed from {h}x{w} to {min_dim}x{min_dim}")

        # Plot the image
        ax.imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")

        # Add mask overlays if requested
        if plot_mask_outlines:
            mask_slice = masks_only[middle_z]
            cell_mask_slice = cell_mask_only[middle_z]

            # Apply same trimming to masks
            if trim_to_square:
                mask_slice = mask_slice[y_start:y_end, x_start:x_end]
                cell_mask_slice = cell_mask_slice[y_start:y_end, x_start:x_end]

            ax.imshow(mask_slice, alpha=0.25, cmap="magma", aspect="equal")
            ax.imshow(cell_mask_slice, alpha=0.5, cmap="hsv", aspect="equal")

        # Set title - include LUT only if adaptive (not fixed for all channels)
        gene_name = channel_to_gene.get(chan, "Unknown")
        if is_fixed_lut:
            # Fixed LUT - show only gene name, LUT will be in suptitle
            if gene_name != "Unknown":
                title = f"{gene_name}"
            else:
                title = f"Ch {chan}"
        ax.set_title(title, fontsize=24)
        # axis off
        ax.axis("off")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.tick_params(labelsize=8)

        # Add row label to the first axis if provided
        if i == 0 and row_label is not None:
            ax.set_ylabel(row_label, fontsize=24, rotation=90, labelpad=10)
        # Add row label to the first axis if provided
        if i == 0 and row_label is not None:
            ax.set_ylabel(row_label, fontsize=24, rotation=90, labelpad=10)
        ax.tick_params(labelsize=8)

        # Add row label to the first axis if provided
        if i == 0 and row_label is not None:
            ax.set_ylabel(row_label, fontsize=22, rotation=90, labelpad=10)
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Adjust layout only if we created our own figure (not using a subfigure)
    # Check if we're working with a subfigure by looking at the type
    is_subfigure = hasattr(fig, "get_gridspec") or "SubFigure" in str(type(fig))

    if not is_subfigure:
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # Adjust top margin for title

    if trim_to_square:
        # Ensure all axes are square
        for ax in axes:
            ax.set_aspect("equal", adjustable="box")
    else:
        # Set aspect to auto if not trimming to square
        for ax in axes:
            ax.set_aspect("auto")

    # No need to remove empty subplots since we use exactly the right number

    # convert origin to int
    origin = tuple(map(int, origin))
    # Add overall title only if we created our own figure (not using a subfigure)
    if not is_subfigure and hasattr(fig, "suptitle"):
        mask_text = "with" if plot_mask_outlines else "without"
        square_text = "square" if trim_to_square else "original"
        fig.suptitle(
            f"All Channels - Cell {cell_id} - Round {round_key} - Centroid {cell_centroid}\n"
            f"Z-plane {middle_z} (global: {origin[0] + middle_z}) - {mask_text} mask overlays - {square_text} aspect",
            fontsize=14,
            fontweight="bold",
        )
    if verbose:
        print("\nPlot completed successfully!")
        print("{'='*60}")

    return fig


def plot_top_spot_count_cells_batch(
    dataset,
    round_key,
    output_dir=None,
    pyramid_level="0",
    volume_percentiles=(10, 90),
    n_top_cells=5,
    auto_vmin_vmax="auto",
    fixed_vmin_vmax=(90, 1200),
    plot_mask_outlines=True,
    trim_to_square=True,
    verbose=True,
):
    """
    Plot and save top N cells for each gene with both auto and fixed intensity scaling.
    Creates combined figures with subfigures showing both scaling methods.

    Parameters:
    -----------
    dataset : HCRDataset
        The HCR dataset object
    round_key : str
        Round identifier (e.g., 'R1')
    output_dir : Path or str, optional
        Output directory. If None, uses 'scratch/{mouse_id}/top_cells_data_combined'
    pyramid_level : str
        Pyramid level for zarr data
    volume_percentiles : tuple
        (min_percentile, max_percentile) for volume filtering
    n_top_cells : int
        Number of top cells to plot per gene
    auto_vmin_vmax : str
        Auto scaling method (default: "auto" for percentile-based)
    fixed_vmin_vmax : tuple
        Fixed intensity range (vmin, vmax)
    plot_mask_outlines : bool
        Whether to overlay segmentation mask outlines
    trim_to_square : bool
        Whether to trim images to square aspect ratio

    Returns:
    --------
    dict
        Dictionary with gene names as keys and lists of saved file paths as values
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if verbose:
        print(f"\n{'='*80}")
        print(f"PLOTTING TOP {n_top_cells} CELLS FOR EACH GENE - ROUND {round_key}")
        print("Combined Auto + Fixed Intensity Scaling")
        print("{'='*80}")

    # Set up output directory
    if output_dir is None:
        mouse_id = getattr(dataset, "mouse_id", "unknown_mouse")
        output_dir = Path("../scratch") / mouse_id / "top_spot_count_cells"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load and filter cell-gene data
    print("\nLoading cell-gene data...")
    mixed_cxg = pd.read_csv(dataset.rounds[round_key].spot_files.unmixed_cxg)
    print(f"Original data shape: {mixed_cxg.shape}")

    # Apply volume filtering
    vol_min = np.percentile(mixed_cxg["volume"], volume_percentiles[0])
    vol_max = np.percentile(mixed_cxg["volume"], volume_percentiles[1])
    filtered_data = mixed_cxg[(mixed_cxg["volume"] >= vol_min) & (mixed_cxg["volume"] <= vol_max)]
    print(
        f"After volume filtering ({volume_percentiles[0]}-{volume_percentiles[1]}th percentile): {filtered_data.shape}"
    )
    print(f"Volume range: {vol_min:.0f} - {vol_max:.0f}")

    # Get top cells for each gene
    print(f"\nFinding top {n_top_cells} cells for each gene...")
    top_cells_data = filtered_data[["gene", "cell_id", "spot_count", "centroid"]].copy()
    top_spot_cells = (
        top_cells_data.groupby("gene").apply(lambda x: x.nlargest(n_top_cells, "spot_count")).reset_index(drop=True)
    )

    # Summary of top cells
    genes = sorted(top_spot_cells["gene"].unique())
    print(f"Genes found: {genes}")
    for gene in genes:
        gene_cells = top_spot_cells[top_spot_cells["gene"] == gene]
        print(
            f"  {gene}: {len(gene_cells)} cells, spot counts: {gene_cells['spot_count'].min()}-{gene_cells['spot_count'].max()}"
        )

    # Plot and save each cell
    saved_files = {}
    total_cells = len(top_spot_cells)

    print(f"\nStarting to plot {total_cells} cells with combined auto/fixed scaling...")

    for idx, (_, row) in enumerate(top_spot_cells.iterrows()):
        gene = row["gene"]
        cell_id = int(row["cell_id"])
        spot_count = int(row["spot_count"])

        # Calculate rank within gene
        gene_cells = top_spot_cells[top_spot_cells["gene"] == gene]
        rank = (top_spot_cells[top_spot_cells["gene"] == gene].index <= idx).sum()

        print(f"\n[{idx+1}/{total_cells}] Plotting {gene} - Cell {cell_id} (spots: {spot_count}, rank: {rank})")

        try:
            # Create combined figure with subfigures
            fig = plt.figure(figsize=(40, 10))  # Wide figure for two rows

            # Create subfigures (2 rows, 1 column) with minimal spacing
            subfigs = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.02)

            # Generate auto-scaled plot (top subfigure)
            print("  Generating auto-scaled plot...")
            try:
                plot_all_channels_cell(
                    dataset=dataset,
                    round_key=round_key,
                    cell_id=cell_id,
                    pyramid_level=pyramid_level,
                    vmin_vmax=auto_vmin_vmax,
                    plot_mask_outlines=False,
                    trim_to_square=trim_to_square,
                    figsize=(20, 5),
                    fig=subfigs[0],  # Pass the subfigure
                )
                auto_success = True

                # Clear any title from the subfigure to avoid conflicts
                if hasattr(subfigs[0], "_suptitle") and subfigs[0]._suptitle:
                    subfigs[0]._suptitle.set_text("")
            except Exception as e:
                print(f"    Error in auto plot: {e}")
                auto_success = False

            # Generate fixed-scaled plot (bottom subfigure)
            print("  Generating fixed-scaled plot...")
            try:
                plot_all_channels_cell(
                    dataset=dataset,
                    round_key=round_key,
                    cell_id=cell_id,
                    pyramid_level=pyramid_level,
                    vmin_vmax=fixed_vmin_vmax,
                    plot_mask_outlines=plot_mask_outlines,
                    trim_to_square=trim_to_square,
                    figsize=(20, 5),
                    fig=subfigs[1],  # Pass the subfigure
                )
                fixed_success = True

                # Clear any title from the subfigure to avoid conflicts
                if hasattr(subfigs[1], "_suptitle") and subfigs[1]._suptitle:
                    subfigs[1]._suptitle.set_text("")
            except Exception as e:
                print(f"    Error in fixed plot: {e}")
                fixed_success = False

            if auto_success and fixed_success:
                # Add overall title
                # mask_text = "with" if plot_mask_outlines else "without"
                # square_text = "square" if trim_to_square else "original"

                # fig.suptitle(f"{gene} - Cell {cell_id} - Round {round_key} - {spot_count} spots (rank {rank})\n"
                #            f"Auto vs Fixed Intensity Scaling - {mask_text} mask overlays - {square_text} aspect",
                #            fontsize=16, fontweight='bold', y=0.95)

                # Add overall title with better positioning
                fig.suptitle(
                    f"{gene} - Cell {cell_id} - Round {round_key} - {spot_count} spots (rank {rank})",
                    fontsize=20,
                    fontweight="bold",
                    y=1.05,
                )  # Position closer to top

                # Ensure proper layout for subfigures
                plt.subplots_adjust(top=0.93, bottom=0.05)  # Adjust margins for better spacing

                # Create filename
                filename = f"{gene}_cell_{cell_id}_spots_{spot_count}_rank_{rank}.png"
                filepath = output_dir / filename

                # Save the figure with improved layout
                fig.savefig(
                    filepath, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.1
                )  # Minimal padding

                # Track saved files
                if gene not in saved_files:
                    saved_files[gene] = []
                saved_files[gene].append(filepath)

                print(f"  ✓ Saved combined plot: {filename}")
            else:
                print(f"  ✗ Failed to generate one or both plots for cell {cell_id}")

            plt.close(fig)  # Close to free memory

        except Exception as e:
            print(f"  ✗ Error plotting cell {cell_id}: {e}")
            continue

    # Summary
    print("\n{'='*80}")
    print("COMBINED PLOTTING COMPLETE!")
    print("{'='*80}")
    print(f"Output directory: {output_dir}")
    print("Files saved by gene:")

    total_saved = 0
    for gene in sorted(saved_files.keys()):
        files = saved_files[gene]
        print(f"  {gene}: {len(files)} combined files")
        total_saved += len(files)

    print(f"\nTotal combined files saved: {total_saved}/{total_cells}")

    # Create a summary file
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("Top Cells Combined Analysis Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset: {getattr(dataset, 'mouse_id', 'unknown')}\n")
        f.write(f"Round: {round_key}\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"Volume filter: {volume_percentiles[0]}-{volume_percentiles[1]}th percentile ({vol_min:.0f}-{vol_max:.0f})\n"
        )
        f.write(f"Top cells per gene: {n_top_cells}\n")
        f.write(f"Auto scaling: {auto_vmin_vmax}\n")
        f.write(f"Fixed scaling: {fixed_vmin_vmax}\n")
        f.write(f"Total combined files saved: {total_saved}\n\n")

        f.write("Files by gene:\n")
        for gene in sorted(saved_files.keys()):
            f.write(f"  {gene}:\n")
            for filepath in saved_files[gene]:
                f.write(f"    {filepath.name}\n")

    print(f"Summary saved to: {summary_file}")

    return saved_files
