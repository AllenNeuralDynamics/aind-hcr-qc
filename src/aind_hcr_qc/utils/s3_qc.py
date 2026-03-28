"""S3 utilities for QC plot management.

Plots are stored under a canonical prefix:
    s3://{bucket}/ctl/hcr/qc/{mouse_id}/{plot_type}.png

A JSON sidecar lives alongside each PNG:
    s3://{bucket}/ctl/hcr/qc/{mouse_id}/{plot_type}.json
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import boto3
import matplotlib.pyplot as plt
from botocore.exceptions import ClientError

import aind_hcr_qc

QC_S3_BUCKET: str = "aind-scratch-data"
QC_S3_PREFIX: str = "ctl/hcr/qc"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_s3_key(
    mouse_id: str,
    plot_type: str,
    ext: str,
    prefix: str = QC_S3_PREFIX,
) -> str:
    """Return the canonical S3 key for a plot artifact.

    Parameters
    ----------
    mouse_id:
        Subject / mouse identifier, e.g. ``"782149"``.
    plot_type:
        Short identifier for the plot, e.g. ``"intensity_violins_round_chan"``.
    ext:
        File extension without the leading dot, e.g. ``"png"`` or ``"json"``.
    prefix:
        S3 key prefix (no trailing slash).
    """
    return f"{prefix}/{mouse_id}/{plot_type}.{ext}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_plot_exists(
    bucket: str,
    mouse_id: str,
    plot_type: str,
    prefix: str = QC_S3_PREFIX,
) -> bool:
    """Return ``True`` if the PNG for this plot already exists on S3.

    Uses a lightweight ``HEAD`` request — no data is downloaded.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    mouse_id:
        Subject identifier.
    plot_type:
        Short plot identifier.
    prefix:
        S3 key prefix (no trailing slash).
    """
    s3 = boto3.client("s3")
    key = _get_s3_key(mouse_id, plot_type, "png", prefix)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code in ("404", "NoSuchKey"):
            return False
        raise


def upload_plot(
    fig: plt.Figure | None,
    bucket: str,
    mouse_id: str,
    plot_type: str,
    metadata: dict[str, Any] | None = None,
    dpi: int = 300,
    prefix: str = QC_S3_PREFIX,
) -> str:
    """Upload a matplotlib figure to S3 and write a JSON sidecar alongside it.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.  Pass ``None`` to use ``plt.gcf()``.
    bucket:
        S3 bucket name.
    mouse_id:
        Subject identifier.
    plot_type:
        Short plot identifier used in the S3 key.
    metadata:
        Extra fields to merge into the sidecar JSON, e.g.
        ``{"plot_kwargs": {...}, "source_assets": {"rounds": {...}, "pairwise_unmixing": "..."}}``.
    dpi:
        Resolution for the PNG.
    prefix:
        S3 key prefix (no trailing slash).

    Returns
    -------
    str
        Full ``s3://bucket/key`` URI of the uploaded PNG.
    """
    if fig is None:
        fig = plt.gcf()

    s3 = boto3.client("s3")
    png_key = _get_s3_key(mouse_id, plot_type, "png", prefix)
    json_key = _get_s3_key(mouse_id, plot_type, "json", prefix)

    # --- upload PNG in-memory (no temp files) ---
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    s3.put_object(
        Bucket=bucket,
        Key=png_key,
        Body=buf.getvalue(),
        ContentType="image/png",
    )

    # --- build and upload sidecar JSON ---
    sidecar = _build_sidecar(bucket, mouse_id, plot_type, png_key, metadata)
    s3.put_object(
        Bucket=bucket,
        Key=json_key,
        Body=json.dumps(sidecar, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    s3_uri = f"s3://{bucket}/{png_key}"
    print(f"Uploaded QC plot → {s3_uri}")
    return s3_uri


def download_plot(
    bucket: str,
    mouse_id: str,
    plot_type: str,
    output_path: str | None = None,
    prefix: str = QC_S3_PREFIX,
    display: bool = False,
):
    """Download a QC plot PNG from S3 to the local filesystem.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    mouse_id:
        Subject identifier.
    plot_type:
        Short plot identifier.
    output_path:
        Local path to save the PNG.  Defaults to
        ``/tmp/qc_{mouse_id}_{plot_type}.png``.
    prefix:
        S3 key prefix (no trailing slash).
    display:
        If ``True`` and running inside a Jupyter notebook, display the image
        inline after downloading.

    Returns
    -------
    Path
        Absolute path to the downloaded file.
    """
    from pathlib import Path

    key = _get_s3_key(mouse_id, plot_type, "png", prefix)

    if output_path is None:
        output_path = f"/tmp/qc_{mouse_id}_{plot_type}.png"

    dest = Path(output_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    s3.download_file(Bucket=bucket, Key=key, Filename=str(dest))
    print(f"Downloaded s3://{bucket}/{key} → {dest}")

    if display:
        try:
            from IPython.display import Image as IPImage, display as ipy_display
            ipy_display(IPImage(filename=str(dest)))
        except ImportError:
            print("IPython not available — open the file manually.")

    return dest


def load_plot_metadata(
    bucket: str,
    mouse_id: str,
    plot_type: str,
    prefix: str = QC_S3_PREFIX,
) -> dict | None:
    """Fetch the JSON sidecar for an existing plot.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    mouse_id:
        Subject identifier.
    plot_type:
        Short plot identifier.
    prefix:
        S3 key prefix (no trailing slash).

    Returns
    -------
    dict or None
        Parsed sidecar contents, or ``None`` if the file does not exist.
    """
    s3 = boto3.client("s3")
    key = _get_s3_key(mouse_id, plot_type, "json", prefix)
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read())
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code in ("404", "NoSuchKey"):
            return None
        raise


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _build_sidecar(
    bucket: str,
    mouse_id: str,
    plot_type: str,
    png_key: str,
    metadata: dict[str, Any] | None,
) -> dict:
    """Assemble the sidecar dictionary with standard fields.

    Standard fields
    ---------------
    created_at
        ISO 8601 UTC timestamp of generation.
    mouse_id
        Subject identifier.
    plot_type
        Short human-readable plot identifier.
    s3_key
        Full ``s3://`` URI of the PNG.
    aind_hcr_qc_version
        Package version for reproducibility.

    Caller-supplied ``metadata`` is merged in after the standard fields and
    may extend or override anything except ``created_at``.  Recommended extra
    fields:

    plot_kwargs : dict
        Key parameters that affect plot output (e.g. ``intensity_threshold``,
        ``order``, ``n_sample``).
    source_assets : dict
        All upstream assets that contributed to the plot, e.g.::

            {
                "rounds": {"R1": "HCR_782149_...", "R2": "..."},
                "pairwise_unmixing": "HCR_782149_pairwise-unmixing_...",
            }
    """
    sidecar: dict[str, Any] = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "mouse_id": mouse_id,
        "plot_type": plot_type,
        "s3_key": f"s3://{bucket}/{png_key}",
        "aind_hcr_qc_version": aind_hcr_qc.__version__,
    }
    if metadata:
        sidecar.update(metadata)
    return sidecar
