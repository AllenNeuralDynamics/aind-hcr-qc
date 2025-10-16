# -*- coding: utf-8 -*-
"""
Plotting functions for visualizing single cell expression data across multiple HCR rounds.
"""
from typing import List
import numpy as np

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from aind_hcr_data_loader.hcr_dataset import HCRDataset
from aind_hcr_qc.utils.utils import saveable_plot

import aind_hcr_qc.io.zarr_data as zarr_data

# -------------------------------------------------------------------------------------------------
# Multi Round
# -------------------------------------------------------------------------------------------------


@saveable_plot()
def plot_single_cell_expression_all_rounds(
    plot_cell_id: int, dataset: HCRDataset, pyramid_level: str = "0", rounds: List[str] = None, vmin_vmax = "auto", verbose: bool = False,
    linear_unmix_matrix=None
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

    # Create a single parent figure
    fig = plt.figure(figsize=(20, 5 * len(rounds)))

    # Create a GridSpec layout
    gs = gridspec.GridSpec(
        len(rounds), 1, figure=fig, hspace=0.05, wspace=0, top=0.95, bottom=0.05, left=0.05, right=0.95
    )

    # Create subfigures for each round
    for i, round_n in enumerate(rounds):
        # Create a subfigure from the gridspec
        subfig = fig.add_subfigure(gs[i, :])

        try:
            # Plot directly on the subfigure
            plot_all_channels_cell(
                dataset=dataset,
                round_key=round_n,
                cell_id=plot_cell_id,
                pyramid_level=pyramid_level,
                vmin_vmax=vmin_vmax,  # Use 5th-95th percentile
                plot_mask_outlines=True,
                trim_to_square=True,  # Default - trim to square
                figsize=None,  # Wide figure for single row
                verbose=verbose,
                fig=subfig,  # Pass the subfigure
                linear_unmix_matrix=linear_unmix_matrix
            )
        except Exception as e:
            print(f"Error plotting round {round_n} for cell {plot_cell_id}: {e}")
            # add a placeholder for the subfigure, say an empty plot
            subfig.add_subplot(111).text(0.5, 0.5, f"Error: {e}", fontsize=12, ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            continue

        # Add title to the subfigure
        subfig.suptitle(f"Round {round_n}", fontsize=16, y=0.98)

    # Add overall title
    fig.suptitle(f"Cell ID {plot_cell_id}", fontsize=18, y=1.05)
    # plt.subplots_adjust(
    #     top=0.95,
    #     bottom=0.05,
    #     left=0.02,
    #     right=0.98,
    #     hspace=0.1  # Small height spacing
    # )

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
    try:
        channel_gene_table = dataset.create_channel_gene_table(spots_only=False)
        round_genes = channel_gene_table[channel_gene_table["Round"] == round_key]
        channel_to_gene = dict(zip(round_genes["Channel"].astype(str), round_genes["Gene"]))
        if verbose:
            print(f"Channel-gene mapping: {channel_to_gene}")
    except Exception as e:
        print(f"Warning: Could not load gene mapping: {e}")
        channel_to_gene = {}

    # Load cell info and segmentation data

    print("\nLoading cell info and segmentation data...")
    # cell_info_df = dataset.get_cell_info(source="segmentation")
    cell_info_df = dataset.rounds[round_key].get_cell_info(source="mixed_cxg")
    print(cell_info_df.describe())
    cell_info_array = cell_info_df[["z_centroid", "y_centroid", "x_centroid", "cell_id"]].to_numpy()
    segmentation_zarr = dataset.load_segmentation_mask(round_key, pyramid_level)

    # Get reference channel for segmentation overlay (usually 405)
    ref_channel = "405" if "405" in channels else channels_sorted[0]
    print(f"Using reference channel {ref_channel} for segmentation overlay")

    # Extract cell volume using reference channel
    print(f"Extracting cell volume for cell {cell_id}...")
    ref_zarr = dataset.load_zarr_channel(round_key, ref_channel, data_type="fused", pyramid_level=pyramid_level)
    print(f"Reference channel shape: {ref_zarr.shape}")

    seg_crop, img_crop, masks_only, cell_mask_only, origin, z_planes, x_planes = zarr_data.extract_cell_volume(
        segmentation_zarr, ref_zarr, cell_info_array, cell_id, num_planes=num_planes, plot_buffer=plot_buffer
    )

    cell_centroid = cell_info_array[cell_info_array[:, -1] == cell_id, :-1][0]
    print(f"Cell centroid (z, y, x): {cell_centroid}")

    print(f"Cell crop shape: {seg_crop.shape}")
    print(f"Origin: {origin}")
    print(f"Z-planes: {z_planes}")

    # Load all channel data
    print("\nLoading channel data...")
    channel_arrays = {}
    for chan in channels_sorted:
        try:
            chan_zarr = dataset.load_zarr_channel(round_key, chan, data_type="fused", pyramid_level=pyramid_level)
            # Crop the channel data to match segmentation crop
            z0, y0, x0 = origin
            z1, y1, x1 = z0 + seg_crop.shape[0], y0 + seg_crop.shape[1], x0 + seg_crop.shape[2]
            chan_crop = np.asarray(chan_zarr[0, 0, z0:z1, y0:y1, x0:x1])
            channel_arrays[chan] = chan_crop
            print(f"  Channel {chan}: loaded, shape {chan_crop.shape}")
        except Exception as e:
            print(f"  Channel {chan}: failed to load - {e}")
            continue
    if verbose:
        print(f"Successfully loaded {len(channel_arrays)} channels")

    if linear_unmix_matrix is not None:
        chan_names = list(channel_arrays.keys())
        img_arrays = list(channel_arrays.values())
        # if channel =405, remove
        if "405" in chan_names:
            idx_405 = chan_names.index("405")
            chan_names.pop(idx_405)
            img_arrays.pop(idx_405)
            print("Removing channel 405 for linear unmixing")
        img_stack = np.stack(img_arrays, axis=-1)  # shape (Y, X, C)
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
            print(f"  Channel {chan}: unmixed")
        print("Linear unmixing completed.")
    # -------------------------------------------------------------------------------------------------


    # Calculate figure layout - single row
    n_channels = len(channel_arrays)
    if n_channels == 0:
        print("No channels loaded successfully!")
        return None

    cols = n_channels  # All channels in one row
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
        axes = []
        for i in range(n_channels):
            ax = fig.add_subplot(rows, cols, i + 1)
            axes.append(ax)

    # Select middle z-plane for display
    middle_z = z_planes[len(z_planes) // 2]
    print(f"Plotting middle z-plane: {middle_z} (global z: {origin[0] + middle_z})")

    # Plot each channel
    for i, chan in enumerate(channel_arrays.keys()):
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
        elif isinstance(vmin_vmax, (tuple, list)) and len(vmin_vmax) == 2:
            vmin, vmax = vmin_vmax
        else:
            vmin, vmax = chan_data.min(), chan_data.max()

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

        # Set title with gene name if available
        gene_name = channel_to_gene.get(chan, "Unknown")
        title = f"Ch {chan}"
        if gene_name != "Unknown":
            title += f" ({gene_name})"
        title += f"   \nLUT=[{vmin:.0f}-{vmax:.0f}]"

        ax.set_title(title, fontsize=14, fontweight="bold")
        # axis off
        ax.axis("off")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.tick_params(labelsize=8)

    # Adjust layout only if we created our own figure (not using a subfigure)
    # Check if we're working with a subfigure by looking at the type
    is_subfigure = hasattr(fig, "get_gridspec") or "SubFigure" in str(type(fig))

    if not is_subfigure:
        plt.tight_layout()
        plt.subplots_adjust(top=0.80)  # Adjust top margin for title

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
