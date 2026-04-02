"""Integration tests for aind_hcr_qc.utils.s3_qc.

These tests hit the real S3 bucket (aind-scratch-data) under an isolated
test prefix so production QC data is never touched.  They require valid AWS
credentials to be present in the environment (e.g. via an instance role or
~/.aws/credentials) and are skipped automatically when credentials are absent.

Run with:
    pytest tests/test_s3_qc.py -v
"""

import unittest

import boto3
import matplotlib
import matplotlib.pyplot as plt
from botocore.exceptions import NoCredentialsError, ClientError

matplotlib.use("Agg")  # headless — no display needed

from aind_hcr_qc.utils.s3_qc import (
    QC_S3_BUCKET,
    _get_s3_key,
    check_plot_exists,
    download_plot,
    load_plot_metadata,
    upload_plot,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
_TEST_PREFIX = "ctl/hcr/qc/_test"
_MOUSE_ID = "test_mouse_000"
_PLOT_TYPE = "dummy_violin_test"


def _has_s3_credentials() -> bool:
    """Return True if AWS credentials are resolvable."""
    try:
        boto3.client("s3").list_buckets()
        return True
    except (NoCredentialsError, ClientError):
        return False


def _make_dummy_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_title("S3 QC smoke-test — dummy violin")
    ax.set_xlabel("gene")
    ax.set_ylabel("intensity")
    ax.violinplot([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    return fig


def _delete_test_objects():
    """Remove both test artifacts from S3 (best-effort cleanup)."""
    s3 = boto3.client("s3")
    for ext in ("png", "json"):
        key = _get_s3_key(_MOUSE_ID, _PLOT_TYPE, ext, _TEST_PREFIX)
        try:
            s3.delete_object(Bucket=QC_S3_BUCKET, Key=key)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------
@unittest.skipUnless(_has_s3_credentials(), "No AWS credentials available — skipping S3 tests")
class TestS3QcUtils(unittest.TestCase):
    """End-to-end tests for the S3 QC plot management utilities."""

    @classmethod
    def setUpClass(cls):
        """Ensure the test prefix is clean before the suite runs."""
        _delete_test_objects()

    @classmethod
    def tearDownClass(cls):
        """Remove test artifacts from S3 after the suite finishes."""
        _delete_test_objects()

    # ------------------------------------------------------------------
    # check_plot_exists
    # ------------------------------------------------------------------

    def test_check_plot_exists_returns_false_before_upload(self):
        exists = check_plot_exists(QC_S3_BUCKET, _MOUSE_ID, _PLOT_TYPE, prefix=_TEST_PREFIX)
        self.assertFalse(exists)

    # ------------------------------------------------------------------
    # upload_plot
    # ------------------------------------------------------------------

    def test_upload_plot_returns_correct_uri(self):
        fig = _make_dummy_figure()
        try:
            uri = upload_plot(
                fig=fig,
                bucket=QC_S3_BUCKET,
                mouse_id=_MOUSE_ID,
                plot_type=_PLOT_TYPE,
                metadata={
                    "plot_kwargs": {"intensity_threshold": 25.0, "order": "round_chan"},
                    "source_assets": {
                        "rounds": {"R1": "HCR_test_R1", "R2": "HCR_test_R2"},
                        "pairwise_unmixing": "HCR_test_pairwise",
                    },
                },
                dpi=72,
                prefix=_TEST_PREFIX,
            )
        finally:
            plt.close(fig)

        expected = f"s3://{QC_S3_BUCKET}/{_TEST_PREFIX}/{_MOUSE_ID}/{_PLOT_TYPE}.png"
        self.assertEqual(uri, expected)

    # ------------------------------------------------------------------
    # check_plot_exists (after upload)
    # ------------------------------------------------------------------

    def test_check_plot_exists_returns_true_after_upload(self):
        # Depends on upload having run — ordering enforced alphabetically by
        # method name: "b_upload" < "c_check_after".  Use explicit ordering
        # via a helper that uploads if needed.
        if not check_plot_exists(QC_S3_BUCKET, _MOUSE_ID, _PLOT_TYPE, prefix=_TEST_PREFIX):
            fig = _make_dummy_figure()
            try:
                upload_plot(fig=fig, bucket=QC_S3_BUCKET, mouse_id=_MOUSE_ID,
                            plot_type=_PLOT_TYPE, dpi=72, prefix=_TEST_PREFIX)
            finally:
                plt.close(fig)

        exists = check_plot_exists(QC_S3_BUCKET, _MOUSE_ID, _PLOT_TYPE, prefix=_TEST_PREFIX)
        self.assertTrue(exists)

    # ------------------------------------------------------------------
    # load_plot_metadata
    # ------------------------------------------------------------------

    def test_load_plot_metadata_returns_valid_sidecar(self):
        # Always upload fresh so this test owns its sidecar contents.
        _delete_test_objects()
        fig = _make_dummy_figure()
        try:
            upload_plot(
                fig=fig,
                bucket=QC_S3_BUCKET,
                mouse_id=_MOUSE_ID,
                plot_type=_PLOT_TYPE,
                metadata={
                    "plot_kwargs": {"intensity_threshold": 25.0},
                    "source_assets": {"rounds": {"R1": "HCR_test_R1"}},
                },
                dpi=72,
                prefix=_TEST_PREFIX,
            )
        finally:
            plt.close(fig)

        meta = load_plot_metadata(QC_S3_BUCKET, _MOUSE_ID, _PLOT_TYPE, prefix=_TEST_PREFIX)

        self.assertIsNotNone(meta)
        for key in ("created_at", "mouse_id", "plot_type", "s3_key",
                    "aind_hcr_qc_version", "plot_kwargs", "source_assets"):
            self.assertIn(key, meta, f"Sidecar missing key: {key!r}")
        self.assertEqual(meta["mouse_id"], _MOUSE_ID)
        self.assertEqual(meta["plot_type"], _PLOT_TYPE)
        self.assertIsInstance(meta["source_assets"]["rounds"], dict)

    def test_load_plot_metadata_returns_none_for_missing_object(self):
        meta = load_plot_metadata(QC_S3_BUCKET, "no_such_mouse", "no_such_plot",
                                  prefix=_TEST_PREFIX)
        self.assertIsNone(meta)

    # ------------------------------------------------------------------
    # download_plot
    # ------------------------------------------------------------------

    def test_download_plot_saves_file_locally(self):
        import os

        if not check_plot_exists(QC_S3_BUCKET, _MOUSE_ID, _PLOT_TYPE, prefix=_TEST_PREFIX):
            fig = _make_dummy_figure()
            try:
                upload_plot(fig=fig, bucket=QC_S3_BUCKET, mouse_id=_MOUSE_ID,
                            plot_type=_PLOT_TYPE, dpi=72, prefix=_TEST_PREFIX)
            finally:
                plt.close(fig)

        dest = download_plot(
            QC_S3_BUCKET, _MOUSE_ID, _PLOT_TYPE,
            output_path="/tmp/test_s3_qc_download.png",
            prefix=_TEST_PREFIX,
        )

        self.assertTrue(dest.exists())
        self.assertGreater(os.path.getsize(dest), 0)


if __name__ == "__main__":
    unittest.main()
