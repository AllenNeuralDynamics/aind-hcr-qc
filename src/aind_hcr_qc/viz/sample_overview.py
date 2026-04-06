"""Tile overview plot for a single HCR round.

Generates a Z-max projection of a fused channel at a coarse pyramid level,
providing a quick sample-coverage check for each round.  The fused zarr
from the processed asset is used directly via the ``HCRRound`` API — no raw
tile stitching or custom S3 access code is needed.
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np

# Regex that matches the ``_processed_<date>_<time>`` suffix appended to
# processed asset names, e.g. ``_processed_2025-07-21_17-35-04``.
_PROCESSED_SUFFIX_RE = re.compile(r"_processed_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def raw_asset_name_from_processed(processed_name: str) -> str:
    """Return the raw asset name by stripping the ``_processed_*`` suffix.

    Parameters
    ----------
    processed_name:
        Full processed asset folder name, e.g.
        ``"HCR_755252_2025-07-10_13-00-00_processed_2025-07-21_17-35-04"``.

    Returns
    -------
    str
        Raw asset name, e.g. ``"HCR_755252_2025-07-10_13-00-00"``.
    """
    return _PROCESSED_SUFFIX_RE.sub("", processed_name)


def plot_tile_overview(
    hcr_round,
    channel: str = "405",
    pyramid_level: int = 3,
    vmax: int | None = None,
) -> plt.Figure:
    """Generate a Z-max projection tile overview for one HCR round.

    Loads the fused zarr at *pyramid_level* — a coarser level (3–4) loads in
    seconds; a finer level (0–1) gives full resolution.  A max-projection
    along Z is computed lazily via dask and rendered as a grayscale image.

    Parameters
    ----------
    hcr_round:
        An ``HCRRound`` instance (from ``aind_hcr_data_loader``).
    channel:
        Channel wavelength string, e.g. ``"405"``.
    pyramid_level:
        Zarr resolution level to load (0 = full resolution, higher = coarser).
    vmax:
        Upper display limit for ``imshow``.  When ``None`` the 99.5th
        percentile of the projection is used.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = hcr_round.load_zarr_channel(channel, data_type="fused", pyramid_level=pyramid_level)

    # OME-Zarr levels may carry leading singleton dims (t, c); squeeze them.
    data = data.squeeze()
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3-D array (Z, Y, X) after squeeze, got shape {data.shape} "
            f"for round {hcr_round.round_key} ch{channel} level {pyramid_level}"
        )

    proj = data.max(axis=0).compute().astype(float)

    if vmax is None:
        vmax = float(np.percentile(proj, 99.5))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(proj, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{hcr_round.name}  |  ch{channel}  |  level {pyramid_level}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    dataset_name = 'HCR_747107_2025-02-21_13-00-00'
    bucket = 'aind-open-data'

    process_microscopy_data(dataset_name, bucket)