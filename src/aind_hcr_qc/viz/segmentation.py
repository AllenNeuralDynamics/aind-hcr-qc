"""Segmentation"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

# import gene_plotter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from aind_hcr_data_loader.hcr_dataset import HCRDataset

import aind_hcr_qc.io.zarr_data as zarr_data

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# PLOTTING PRIMITIVES
# -----------------------------------------------------------------------------

# Constants for colours
_CHANNEL_COLORS = ["green", "blue", "orange", "red", "purple"]


def plot_cell_volume_and_diameter(
    cell_df, q1=None, q2=None, resolution={"x": 0.24, "y": 0.24, "z": 1.0}, figsize=(10, 4), fig=None, axes=None
):
    """
    Plot cell volume distribution and estimated cell diameter based on volume.

    Parameters:
    -----------
    cell_df : pandas.DataFrame
        DataFrame containing cell information with a 'volume' column
    q1, q2 : float
        Quantiles for filtering and showing on plots (default: 0.05, 0.95)
    resolution : dict
        Resolution in microns per voxel for x, y, z dimensions
    figsize : tuple
        Figure size for the plot
    fig : matplotlib figure, optional
        Existing figure to use for plotting, if None a new figure is created
    axes : list of matplotlib axes, optional
        Existing axes to use for plotting, if None new axes are created

    Returns:
    --------
    fig : matplotlib figure
        The figure object containing the plots
    diameters : numpy.ndarray
        The calculated cell diameters in microns
    """
    # Removed redundant import of matplotlib.pyplot

    # Need to import seaborn for KDE plot
    # Create figure with two subplots
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1, ax2 = axes

    # Plot 1: Cell volume distribution
    ax1.hist(cell_df["volume"], bins=75, color="lightblue", edgecolor="black")

    if q1 is not None:
        ax1.axvline(
            cell_df["volume"].quantile(q1), color="red", linestyle="dashed", linewidth=1, label=f"min_per: ({q1})"
        )

    if q2 is not None:
        ax1.axvline(
            cell_df["volume"].quantile(q2), color="green", linestyle="dashed", linewidth=1, label=f"max_per: ({q2})"
        )
    ax1.axvline(
        cell_df["volume"].median(),
        color="black",
        linestyle="dashed",
        linewidth=1,
        label=f'median: {int(cell_df["volume"].median())} voxels',
    )
    n_rois = cell_df.shape[0]
    ax1.set_title(f"Cell Volume Distribution (n_rois={n_rois})")
    ax1.set_xlabel("Cell Volume (voxels)")
    ax1.set_ylabel("Frequency")
    # sci notation for x axis
    ax1.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax1.legend()

    # Convert volume from voxels to cubic microns
    volume_conversion = resolution["x"] * resolution["y"] * resolution["z"]
    volumes_um3 = cell_df["volume"] * volume_conversion

    # Calculate diameter assuming spherical cells (d = 2 * (3V/4π)^(1/3))
    diameters = 2 * ((3 * volumes_um3) / (4 * np.pi)) ** (1 / 3)

    # Plot 2: Estimated cell diameter
    sns.kdeplot(diameters, ax=ax2, fill=True, color="skyblue")
    if q1 is not None:
        ax2.axvline(diameters.quantile(q1), color="red", linestyle="dashed", linewidth=1, label=f"min_per: ({q1})")
    if q2 is not None:
        ax2.axvline(diameters.quantile(q2), color="green", linestyle="dashed", linewidth=1, label=f"max_per: ({q2})")
    ax2.axvline(
        diameters.median(), color="black", linestyle="dashed", linewidth=1, label=f"median: {diameters.median():.2f} µm"
    )

    ax2.set_title("Estimated Cell Diameter (assuming sphere)")
    ax2.set_xlabel("Estimated Diameter (µm)")
    ax2.set_ylabel("Density")
    ax2.legend()

    plt.tight_layout()

    print_stats = True
    if print_stats:
        print("Cell diameter statistics (µm):")
        print(f"Min: {diameters.min():.2f}")
        print(f"Max: {diameters.max():.2f}")
        print(f"Mean: {diameters.mean():.2f}")
        print(f"Median: {diameters.median():.2f}")
        print(f"5th percentile: {diameters.quantile(0.05):.2f}")
        print(f"95th percentile: {diameters.quantile(0.95):.2f}")
    return fig, diameters


def filter_cell_info(cell_info, q1=0.2, q2=0.95):
    # TODO: add to HCRDataset and make smarter
    n_cells = cell_info.shape[0]
    cell_info = cell_info[
        (cell_info["volume"] > cell_info["volume"].quantile(q1))
        & (cell_info["volume"] < cell_info["volume"].quantile(q2))
    ]
    print(f"Kept {cell_info.shape[0]} cells out of {n_cells} total cells, based on volume quantiles {q1} and {q2}.")
    return cell_info


def fig_centroids_filtered(cell_info, q1=0.2, q2=0.95, save=False, output_dir=None):
    """
    Create a figure with two subplots showing cell volume and diameter distributions
    before and after filtering.

    Parameters:
    -----------
    cell_info : pandas.DataFrame
        DataFrame containing cell information with volume data
    q1, q2 : float
        Lower and upper quantile thresholds for volume filtering
    save : bool
        Whether to save the figure to disk
    output_dir : Path or str
        Directory to save the figure if save=True

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The combined figure
    cell_df_filt : pandas.DataFrame
        The filtered cell dataframe
    """
    # Create the filtered dataframe
    cell_df_filt = filter_cell_info(cell_info, q1=q1, q2=q2)

    # Create a figure with subfigures
    fig = plt.figure(figsize=(12, 8))
    subfigs = fig.subfigures(2, 1, wspace=0.07, hspace=0.5)

    # Original data plot
    ax1 = subfigs[0].subplots(1, 2)
    _, diameters = plot_cell_volume_and_diameter(cell_info, q1=q1, q2=q2, fig=subfigs[0], axes=ax1)
    subfigs[0].suptitle("Original Cell Distribution", fontsize=14, y=1.25)

    # Filtered data plot
    ax2 = subfigs[1].subplots(1, 2)
    _, diameters = plot_cell_volume_and_diameter(cell_df_filt, q1=None, q2=None, fig=subfigs[1], axes=ax2)
    subfigs[1].suptitle("Filtered Cell Distribution", fontsize=14, y=1.25)

    # Add overall title
    # fig.suptitle(f"Cell Volume and Diameter Distributions (filtered: q1={q1}, q2={q2})", fontsize=16)

    # Save figure if requested
    if save and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        fig_path = output_dir / f"cell_volume_diameter_q{q1}_{q2}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {fig_path}")

    return fig, cell_df_filt


def plot_centroids(
    df,
    orientation="XY",
    n_samples=None,
    color_col=None,
    cmap="Blues",
    size=4,
    clip_range=(None, None),
    xlims=(None, None),
    ylims=(None, None),
    random_state=42,
    ax=None,
):
    """
    Plot cell centroids in different orientations

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing centroid coordinates
    orientation : str
        One of 'XY', 'ZX', or 'ZY'
    n_samples : int
        Number of random cells to plot
    random_state : int
        Random seed for reproducibility
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure and axes.
    """
    # Sample random cells
    if n_samples is not None:
        df = df.sample(n=n_samples, random_state=random_state)

    # map to common centroid name for cols
    df = df.rename(columns={'centroid_x': 'x_centroid', 'centroid_y': 'y_centroid', 'centroid_z': 'z_centroid'})

    # if clip_range is specified, clip the color_col
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

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    # use color col to set colorbar of scatter values
    if color_col is not None:
        scatter = ax.scatter(df[x_coord], df[y_coord], alpha=0.5, c=df[color_col], cmap=cmap, s=size)
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_col, rotation=270, labelpad=15)
    else:
        ax.scatter(df[x_coord], df[y_coord], alpha=0.3, s=size)
    if xlims[0] is not None or xlims[1] is not None:
        ax.set_xlim(xlims[0], xlims[1])
    else:
        ax.set_xlim(df[x_coord].min() - 10, df[x_coord].max() + 10)
    if ylims[0] is not None or ylims[1] is not None:
        ax.set_ylim(ylims[0], ylims[1])
    else:
        ax.set_ylim(df[y_coord].min() - 10, df[y_coord].max() + 10)
    #ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{x_coord}")
    ax.set_ylabel(f"{y_coord}")
    ax.set_title(f"Cell Centroids - {plane} (n={n_samples if n_samples is not None else len(df)})")

    # Reverse y-axis for ZX, ZY, and XY
    if orientation in ["ZX", "ZY", "XY"]:
        ax.invert_yaxis()

    if ax is None:
        plt.show()
    return ax


def fig_cell_centroids_comparison(cell_df, cell_df_filt, orientation="XY", save=False, output_dir=None):
    """
    Plot cell centroids before and after filtering side by side.

    Args:
        cell_df (pd.DataFrame): Original cell information
        cell_df_filt (pd.DataFrame): Filtered cell information
        orientation (str): Orientation of the plot ("XY", "ZX", or "ZY")
        save (bool): Whether to save the plot
        output_dir (Path or str): Directory to save the plot (required if save=True)
    """

    # if xy, 1,2 else 2,1
    n_row = 1 if orientation == "XY" else 2
    n_col = 2 if orientation == "XY" else 1
    fig, (ax1, ax2) = plt.subplots(n_row, n_col, figsize=(8, 8))

    # Define orientation mappings for coords and limits
    orientation_maps = {
        "XY": {"x": "x_centroid", "y": "y_centroid"},
        "ZX": {"x": "x_centroid", "y": "z_centroid"},
        "ZY": {"x": "y_centroid", "y": "z_centroid"},
    }

    if orientation not in orientation_maps:
        raise ValueError("Orientation must be one of 'XY', 'ZX', or 'ZY'")

    x_coord = orientation_maps[orientation]["x"]
    y_coord = orientation_maps[orientation]["y"]

    # get max x and y values for scaling based on the orientation
    max_x = max(cell_df[x_coord].max(), cell_df_filt[x_coord].max())
    min_x = min(cell_df[x_coord].min(), cell_df_filt[x_coord].min())
    max_y = max(cell_df[y_coord].max(), cell_df_filt[y_coord].max())
    min_y = min(cell_df[y_coord].min(), cell_df_filt[y_coord].min())

    # Add some padding to the limits
    padding_x = (max_x - min_x) * 0.05
    padding_y = (max_y - min_y) * 0.05
    xlims = (min_x - padding_x, max_x + padding_x)
    ylims = (min_y - padding_y, max_y + padding_y)

    # Plot original data
    plot_centroids(cell_df, orientation=orientation, color_col=None, ax=ax1, xlims=xlims, ylims=ylims)
    ax1.set_title(f"Original Cells (n={len(cell_df)})")

    # Plot filtered data
    plot_centroids(cell_df_filt, orientation=orientation, color_col=None, ax=ax2, xlims=xlims, ylims=ylims)
    ax2.set_title(f"Filtered Cells (n={len(cell_df_filt)})")

    # ax square
    ax1.set_aspect("auto", adjustable="box")
    ax2.set_aspect("auto", adjustable="box")

    plt.tight_layout()

    if save:
        if output_dir is None:
            raise ValueError("output_dir must be specified when save=True")
        output_path = Path(output_dir) / f"cell_centroids_comparison_{orientation}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig


# def plot_single_cell_segmentation_overview(
#     dataset, round_n, pyramid_level, plot_channel, plot_cell_id, num_planes=10, view="multi"
# ):
#     """
#     Plots the segmentation overview for a given cell.

#     Parameters:
#     dataset (HCRDataset): The dataset object containing segmentation and image data.
#     round_n (str): The round identifier (e.g., 'R1').
#     pyramid_level (str): The pyramid level to use for segmentation and image data.
#     plot_channel (str): The channel to plot (e.g., '405').
#     plot_cell_id (int): The cell ID to plot.
#     num_planes (int): Number of planes to extract for visualization. Default is 10.
#     view (bool): If True, plots multiple views of the cell. Default is True.

#     Returns:
#     None
#     """
#     # Get cell info as numpy array
#     # cell_info_array = dataset.get_cell_info(source="segmentation").to_numpy()
#     cell_info = dataset.rounds[round_n].get_cell_info(source="mixed_cxg")
#     cell_info_array = cell_info[["z_centroid", "y_centroid", "x_centroid", "cell_id"]].to_numpy()
#     # Load segmentation mask and image data
#     segmentation_zarr = dataset.load_segmentation_mask(round_n, pyramid_level)
#     seg_image_zarr = dataset.load_zarr_channel(round_n, plot_channel, data_type="fused", pyramid_level=pyramid_level)

#     # Extract cell volume
#     seg_crop, img_crop, masks_only, cell_mask_only, origin, z_planes, x_planes = gene_plotter.extract_cell_volume(
#         segmentation_zarr, seg_image_zarr, cell_info_array, plot_cell_id, num_planes=num_planes
#     )

#     # find matching cell_id in cell_info_array (last column)
#     centroid = cell_info_array[cell_info_array[:, -1] == plot_cell_id, :3].flatten()
#     if centroid.size == 0:
#         raise ValueError(f"Cell ID {plot_cell_id} not found in cell_info_array.")
#     # Plot segmentation overview
#     gene_plotter.plot_segmentation_overview(
#         seg_crop,
#         img_crop,
#         masks_only,
#         cell_mask_only,
#         plot_cell_id,
#         origin,
#         z_planes,
#         x_planes,
#     )
#     plt.show()
#     return


def plot_segmentation_overview(
    seg_crop: np.ndarray,
    img_crop: np.ndarray,
    masks_only: np.ndarray,
    cell_mask_only: np.ndarray,
    cell_id: int,
    origin: Tuple[int, int, int],
    centroid: Tuple[int, int, int],
    z_planes: np.ndarray,
    x_planes: np.ndarray,
    *,
    delta: int = 3,
    # figsize: Tuple[int, int] = (10, 10),
):
    """Visualise *x–z* and *z–y* slices through the cell.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # calc figure size based on number of planes
    num_planes = max(len(z_planes), len(x_planes))
    figsize = (5, 2 * num_planes)  # width, height
    num_planes = len(z_planes)
    fig, axes = plt.subplots(num_planes, 2, figsize=figsize, constrained_layout=True)

    # --------------------------------------------------------------
    #  x–z planes (left column) – iterate over *x* indices
    # --------------------------------------------------------------
    for i, xp in enumerate(x_planes):
        ax = axes[i, 0]
        img = img_crop[:, :, xp - delta : xp + delta + 1].max(axis=2)
        ax.imshow(img.T, cmap="gray", aspect="auto")
        ax.imshow(masks_only[:, :, xp].T, alpha=0.25, cmap="magma", aspect="auto")
        ax.imshow(cell_mask_only[:, :, xp].T, alpha=0.25, cmap="hsv", aspect="auto")
        ax.set_title(f"x = {origin[2] + xp}")
        ax.set_xlabel("z")
        ax.set_ylabel("y")
        ax.set_xticks([])
        ax.set_yticks([])

    # --------------------------------------------------------------
    #  z–y planes (right column) – iterate over *z* indices
    # --------------------------------------------------------------
    for i, zp in enumerate(z_planes):
        ax = axes[i, 1]
        img = img_crop[zp - delta : zp + delta + 1, :, :].max(axis=0)
        ax.imshow(img, cmap="gray", aspect="auto")
        ax.imshow(masks_only[zp], alpha=0.25, cmap="magma", aspect="auto")
        ax.imshow(cell_mask_only[zp], alpha=0.25, cmap="hsv", aspect="auto")
        ax.set_title(f"z = {origin[0] + zp}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Cell {cell_id} - Centroid {centroid}", y=1.02)
    return fig


def plot_segmentation_overview_single(
    seg_crop: np.ndarray,
    img_crop: np.ndarray,
    masks_only: np.ndarray,
    cell_mask_only: np.ndarray,
    cell_id: int,
    origin: Tuple[int, int, int],
    centroid: Tuple[int, int, int],
    z_planes: np.ndarray,
    x_planes: np.ndarray,
    *,
    delta: int = 3,
):
    """Visualise only *z–y* slices through the cell in a tight layout.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Calculate figure size based on number of z planes
    num_planes = len(z_planes)
    figsize = (num_planes * 2, 2)  # Width, height - compact

    # Create figure with single column
    fig = plt.figure(figsize=figsize)

    # Use gridspec for truly adjacent subplots with no spacing
    gs = fig.add_gridspec(1, num_planes, hspace=0, wspace=0)

    # --------------------------------------------------------------
    #  z–y planes (single column) – iterate over *z* indices
    # --------------------------------------------------------------
    for i, zp in enumerate(z_planes):
        ax = fig.add_subplot(gs[0, i])
        img = img_crop[zp - delta : zp + delta + 1, :, :].max(axis=0)
        # trim whichever axis bigger to make square
        if img.shape[0] > img.shape[1]:
            img = img[:, : img.shape[0]]
        elif img.shape[1] > img.shape[0]:
            img = img[: img.shape[1], :]

        logger.debug(f"img.shape: {img.shape}")
        # ax.imshow(img, cmap="gray", aspect="auto")
        # ax.imshow(masks_only[zp], alpha=0.25, cmap="magma", aspect="auto")
        # ax.imshow(
        #     cell_mask_only[zp], alpha=0.25, cmap="hsv", aspect="auto"
        # )
        # calc percentil for vmin and vmax
        vmin, vmax = np.percentile(img, [0, 99.9])
        ax.imshow(img, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        ax.imshow(masks_only[zp], alpha=0.25, cmap="magma", aspect="auto")
        ax.imshow(cell_mask_only[zp], alpha=0.25, cmap="hsv", aspect="auto")

        # No axes, no ticks, no labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        # Mark z plane in the top left corner
        ax.text(
            5,
            15,  # Pixel coordinates
            f"z={origin[0] + zp}",
            color="white",
            fontsize=12,
            bbox=dict(facecolor="black", alpha=0.6, pad=1),
        )

    # Remove all spacing around the figure
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    return fig


def plot_single_cell_segmentation_overview(
    dataset, round_n, pyramid_level, plot_channel, plot_cell_id, num_planes=10, view="multi"
):
    """
    Plots the segmentation overview for a given cell.

    Parameters:
    dataset (HCRDataset): The dataset object containing segmentation and image data.
    round_n (str): The round identifier (e.g., 'R1').
    pyramid_level (str): The pyramid level to use for segmentation and image data.
    plot_channel (str): The channel to plot (e.g., '405').
    plot_cell_id (int): The cell ID to plot.
    num_planes (int): Number of planes to extract for visualization. Default is 10.
    view (bool): If True, plots multiple views of the cell. Default is True.

    Returns:
    None
    """
    # Get cell info as numpy array
    # cell_info_array = dataset.get_cell_info(source="segmentation").to_numpy()
    cell_info = dataset.rounds[round_n].get_cell_info(source="mixed_cxg")
    cell_info_array = cell_info[["z_centroid", "y_centroid", "x_centroid", "cell_id"]].to_numpy()
    # Load segmentation mask and image data
    segmentation_zarr = dataset.load_segmentation_mask(round_n, pyramid_level)
    seg_image_zarr = dataset.load_zarr_channel(round_n, plot_channel, data_type="fused", pyramid_level=pyramid_level)

    # Extract cell volume
    seg_crop, img_crop, masks_only, cell_mask_only, origin, z_planes, x_planes = zarr_data.extract_cell_volume(
        segmentation_zarr, seg_image_zarr, cell_info_array, plot_cell_id, num_planes=num_planes
    )

    # find matching cell_id in cell_info_array (last column)
    centroid = cell_info_array[cell_info_array[:, -1] == plot_cell_id, :3].flatten()
    if centroid.size == 0:
        raise ValueError(f"Cell ID {plot_cell_id} not found in cell_info_array.")
    # Plot segmentation overview
    if view == "multi":
        fig = plot_segmentation_overview(
            seg_crop,
            img_crop,
            masks_only,
            cell_mask_only,
            plot_cell_id,
            origin,
            centroid,
            z_planes,
            x_planes,
        )
    elif view == "single":
        fig = plot_segmentation_overview_single(
            seg_crop,
            img_crop,
            masks_only,
            cell_mask_only,
            plot_cell_id,
            origin,
            centroid,
            z_planes,
            x_planes,
        )
    return fig



# ---
# Main QC function
# ----


def qc_segmentation(
    dataset: HCRDataset, data_dir=Path("/root/capsule/data"), output_dir=Path("/root/capsule/scratch"), verbose=True
):
    """
    Run segmentation quality control analysis.

    Parameters:
    -----------
    dataset : HCRDataset
        The dataset object containing cell information
    data_dir : Path or str
        Path to data directory containing segmentation masks
    output_dir : Path or str
        Directory to save QC outputs
    verbose : bool
        Enable verbose output
    """

    # call fig_centroids_filtered
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    qc_dir = output_dir / "segmentation_qc"
    qc_dir.mkdir(exist_ok=True, parents=True)

    if verbose:
        print(f"Running segmentation QC for dataset: {dataset}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {qc_dir}")

    # add assert, dataset must have "R1" key
    assert "R1" in dataset.rounds, "Dataset must contain 'R1' round for segmentation QC."
    cell_info = dataset.rounds["R1"].get_cell_info()

    q1 = 0.2
    q2 = 0.95

    _, _ = fig_centroids_filtered(cell_info, q1=q1, q2=q2, save=True, output_dir=qc_dir)

    # --- 2. Plot cell centroids before and after filtering ---

    orientations = ["XY", "ZX", "ZY"]
    for orientation in orientations:
        fig_cell_centroids_comparison(
            cell_info, filter_cell_info(cell_info, q1=q1, q2=q2), orientation=orientation, save=True, output_dir=qc_dir
        )
        if verbose:
            print(f"Centroid comparison plot saved for orientation {orientation} to: {qc_dir}")
    if verbose:
        print(f"Segmentation QC plots saved to: {qc_dir}")
