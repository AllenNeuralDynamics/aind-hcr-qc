# gene_plotter.py
"""Modular re‑implementation of the original `plot_cell_gene_chans` routine.

The file is organised into four layers:

1.  **I/O helpers** – small utilities for loading metadata, metrics and raw
    volumes.
2.  **Geometry helpers** – centroids → bounding boxes → cropped volumes.
3.  **Plotting primitives** – single‑responsibility functions that draw one
    logical element (slice, scatter, etc.).
4.  **Orchestration** – higher‑level builders that assemble complete figures
    and execute the batch loop over cells.

Dependencies
------------
* numpy, pandas, matplotlib, zarr, dask.array, pathlib
* Internal helpers: ``get_sample_data_info`` and ``get_mask_outlines`` –
  assumed to be available in the caller's environment.
* ``segmentation_utils.py`` must be importable from the same package/dir.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import skimage


def get_mask_outlines(label_mask_array, label):
    masks_only = np.full(label_mask_array.shape, np.nan)
    boundary_mask = skimage.segmentation.find_boundaries(label_mask_array, connectivity=1, mode="thick")
    masks_only[boundary_mask] = 1

    lab_only_mask = np.where(label_mask_array == label, label_mask_array, 0)
    lab_mask_only = np.full(lab_only_mask.shape, np.nan)
    boundary_mask = skimage.segmentation.find_boundaries(lab_only_mask, connectivity=1, mode="thick")
    lab_mask_only[boundary_mask] = -1

    return masks_only, lab_mask_only


def get_cell_centroid_voxels(cell_centroids: np.ndarray, cell_id: int) -> np.ndarray:
    """Return ndarray [z, y, x] in voxels (after anisotropy scaling)."""
    idx = np.where(cell_centroids[:, -1] == cell_id)[0][0]
    # return cell_centroids[idx, :-1] * np.array([1, 4, 4]).astype(int)
    return cell_centroids[idx, :-1].astype(int)


def extract_cell_volume(
    segmentation_zarr,
    seg_image_zarr,
    cell_centroids: np.ndarray,
    cell_id: int,
    *,
    num_planes: int = 5,
    plot_buffer: int = 50,
    chuck_shape: Tuple[int, int, int] = (200, 200, 200),
):
    """Extract a buffered sub‑volume around ``cell_id``.

    The helper returns both the cropped segmentation mask and the
    corresponding 405‑nm background, plus pre‑computed overlays that the
    main plotting routine can reuse directly.

    Notes
    -----
    The incoming *cell_centroids* is assumed to store coordinates in the
    *anisotropic voxel* space used by the segmentation.  The ``*[1, 4, 4]``
    scaling from the original code is preserved here.
    """

    # ------------------------------------------------------------------
    # Locate the cell centroid in (z, y, x) voxel space
    # ------------------------------------------------------------------
    centroid = get_cell_centroid_voxels(cell_centroids, cell_id)
    cz, cy, cx = centroid
    print(f"Extracting cell {cell_id} at centroid (z, y, x) = ({cz}, {cy}, {cx})")

    # ------------------------------------------------------------------
    # Chunk the volume
    # ------------------------------------------------------------------
    sz, sy, sx = (centroid - np.array(chuck_shape) / 2).astype(int)

    seg_chunk = segmentation_zarr[
        0,
        0,
        sz : sz + chuck_shape[0],
        sy : sy + chuck_shape[1],
        sx : sx + chuck_shape[2],
    ]
    # print all lables in the chunk
    print(f"Labels in chunk: {np.unique(seg_chunk)}")
    # print slices of all orientations for cell
    print(
        f"z: {sz} to {sz + chuck_shape[0]}, " f"y: {sy} to {sy + chuck_shape[1]}, " f"x: {sx} to {sx + chuck_shape[2]}"
    )
    # print zarr shape
    print(f"zarr shape: {segmentation_zarr.shape}")

    # Local → global bbox of cell within the chunk
    zz, yy, xx = np.where(seg_chunk == cell_id)
    bbox_global = (
        sz + zz.min(),
        sy + yy.min(),
        sx + xx.min(),
        sz + zz.max(),
        sy + yy.max(),
        sx + xx.max(),
    )

    # ------------------------------------------------------------------
    # Final buffered crop (global coordinates)
    # ------------------------------------------------------------------
    gshape = np.array(segmentation_zarr.shape[2:])

    zmin, ymin, xmin = np.maximum(np.array(bbox_global[:3]) - plot_buffer, 0)
    zmax, ymax, xmax = np.minimum(np.array(bbox_global[3:]) + plot_buffer, gshape)

    seg_crop = np.asarray(segmentation_zarr[0, 0, zmin:zmax, ymin:ymax, xmin:xmax])
    img_crop = np.asarray(seg_image_zarr[0, 0, zmin:zmax, ymin:ymax, xmin:xmax])

    # ------------------------------------------------------------------
    # Generate outline masks once – reused for every slice later on
    # ------------------------------------------------------------------
    masks_only, cell_mask_only = get_mask_outlines(seg_crop, cell_id)

    # Choose evenly‑spaced planes to plot
    delta = 3
    zz, yy, xx = np.where(seg_crop == cell_id)
    z_planes = np.linspace(zz.min() + delta, zz.max() - delta, num_planes, dtype=int)
    x_planes = np.linspace(xx.min() + delta, xx.max() - delta, num_planes, dtype=int)

    origin = (zmin, ymin, xmin)

    return (
        seg_crop,
        img_crop,
        masks_only,
        cell_mask_only,
        origin,
        z_planes,
        x_planes,
    )
