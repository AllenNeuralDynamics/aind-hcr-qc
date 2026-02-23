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

def extract_volume_around_point(
    zarr_volume,
    center_point: np.ndarray,
    *,
    buffer_size: int | Tuple[int, int, int] = 50,
    return_origin: bool = True,
):
    """Extract a 3D sub-volume around a point from a Zarr array.
    
    Parameters
    ----------
    zarr_volume : zarr.Array
        Input Zarr volume with shape (C, T, Z, Y, X) or (Z, Y, X)
    center_point : np.ndarray
        Center point as [z, y, x] in voxel coordinates
    buffer_size : int or tuple of 3 ints
        Buffer to apply around the center point. If int, applied uniformly.
        If tuple, applied as (z_buffer, y_buffer, x_buffer)
    return_origin : bool
        If True, return tuple of (volume, origin). Origin is (zmin, ymin, xmin)
        in global coordinates
        
    Returns
    -------
    np.ndarray or tuple
        If return_origin=False: cropped volume as numpy array
        If return_origin=True: (cropped volume, (zmin, ymin, xmin))
    """
    # Handle buffer size
    if isinstance(buffer_size, int):
        buffer = np.array([buffer_size, buffer_size, buffer_size])
    else:
        buffer = np.array(buffer_size)
    
    # Get volume shape (handle both (C, T, Z, Y, X) and (Z, Y, X))
    if len(zarr_volume.shape) == 5:
        vol_shape = np.array(zarr_volume.shape[2:])  # (Z, Y, X)
        has_ct_dims = True
    else:
        vol_shape = np.array(zarr_volume.shape)
        has_ct_dims = False
    
    # Calculate bounding box with clipping to volume boundaries
    center = np.array(center_point).astype(int)
    mins = np.maximum(center - buffer, 0)
    maxs = np.minimum(center + buffer, vol_shape)
    
    zmin, ymin, xmin = mins
    zmax, ymax, xmax = maxs
    
    # Extract volume
    if has_ct_dims:
        volume_crop = np.asarray(zarr_volume[0, 0, zmin:zmax, ymin:ymax, xmin:xmax])
    else:
        volume_crop = np.asarray(zarr_volume[zmin:zmax, ymin:ymax, xmin:xmax])
    
    if return_origin:
        return volume_crop, (zmin, ymin, xmin)
    return volume_crop


def extract_plane_around_point(
    zarr_volume,
    center_point: np.ndarray,
    plane_axis: str = 'z',
    plane_offset: int = 0,
    *,
    buffer_size: int | Tuple[int, int] = 50,
    return_metadata: bool = True,
):
    """Extract a 2D plane around a point from a Zarr array.
    
    Parameters
    ----------
    zarr_volume : zarr.Array
        Input Zarr volume with shape (C, T, Z, Y, X) or (Z, Y, X)
    center_point : np.ndarray
        Center point as [z, y, x] in voxel coordinates
    plane_axis : str
        Axis perpendicular to the plane: 'z', 'y', or 'x'
    plane_offset : int
        Offset from the center point along plane_axis (e.g., +2 = 2 slices above center)
    buffer_size : int or tuple of 2 ints
        Buffer in the two dimensions parallel to the plane
        If int, applied uniformly. If tuple, order depends on plane_axis:
        - 'z': (y_buffer, x_buffer)
        - 'y': (z_buffer, x_buffer)
        - 'x': (z_buffer, y_buffer)
    return_metadata : bool
        If True, return dict with plane, origin, and plane_index
        
    Returns
    -------
    np.ndarray or dict
        If return_metadata=False: 2D plane as numpy array
        If return_metadata=True: dict with keys:
            - 'plane': 2D numpy array
            - 'plane_index': global coordinate of the plane
            - 'origin': (min_dim1, min_dim2) in global coordinates
            - 'plane_axis': which axis was sliced
    """
    # Get volume shape
    if len(zarr_volume.shape) == 5:
        vol_shape = np.array(zarr_volume.shape[2:])  # (Z, Y, X)
        has_ct_dims = True
    else:
        vol_shape = np.array(zarr_volume.shape)
        has_ct_dims = False
    
    # Parse buffer
    if isinstance(buffer_size, int):
        buffer = (buffer_size, buffer_size)
    else:
        buffer = buffer_size
    
    center = np.array(center_point).astype(int)
    
    # Determine plane index and extraction logic based on axis
    if plane_axis.lower() == 'z':
        plane_idx = np.clip(center[0] + plane_offset, 0, vol_shape[0] - 1)
        y_min = max(0, center[1] - buffer[0])
        y_max = min(vol_shape[1], center[1] + buffer[0])
        x_min = max(0, center[2] - buffer[1])
        x_max = min(vol_shape[2], center[2] + buffer[1])
        
        if has_ct_dims:
            plane = np.asarray(zarr_volume[0, 0, plane_idx, y_min:y_max, x_min:x_max])
        else:
            plane = np.asarray(zarr_volume[plane_idx, y_min:y_max, x_min:x_max])
        
        origin = (y_min, x_min)
        
    elif plane_axis.lower() == 'y':
        plane_idx = np.clip(center[1] + plane_offset, 0, vol_shape[1] - 1)
        z_min = max(0, center[0] - buffer[0])
        z_max = min(vol_shape[0], center[0] + buffer[0])
        x_min = max(0, center[2] - buffer[1])
        x_max = min(vol_shape[2], center[2] + buffer[1])
        
        if has_ct_dims:
            plane = np.asarray(zarr_volume[0, 0, z_min:z_max, plane_idx, x_min:x_max])
        else:
            plane = np.asarray(zarr_volume[z_min:z_max, plane_idx, x_min:x_max])
        
        origin = (z_min, x_min)
        
    elif plane_axis.lower() == 'x':
        plane_idx = np.clip(center[2] + plane_offset, 0, vol_shape[2] - 1)
        z_min = max(0, center[0] - buffer[0])
        z_max = min(vol_shape[0], center[0] + buffer[0])
        y_min = max(0, center[1] - buffer[1])
        y_max = min(vol_shape[1], center[1] + buffer[1])
        
        if has_ct_dims:
            plane = np.asarray(zarr_volume[0, 0, z_min:z_max, y_min:y_max, plane_idx])
        else:
            plane = np.asarray(zarr_volume[z_min:z_max, y_min:y_max, plane_idx])
        
        origin = (z_min, y_min)
    else:
        raise ValueError(f"plane_axis must be 'z', 'y', or 'x', got '{plane_axis}'")
    
    if return_metadata:
        return {
            'plane': plane,
            'plane_index': plane_idx,
            'origin': origin,
            'plane_axis': plane_axis.lower()
        }
    return plane