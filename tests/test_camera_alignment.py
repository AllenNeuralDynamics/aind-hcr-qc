"""Tests for camera_alignment plotting and helper functions."""
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import unittest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.testing import assert_allclose
from pathlib import Path

from aind_hcr_qc.camera_alignment import (
    _extract_translation_stats,
    load_tile_metrics,
    plot_translation_distributions,
    plot_translation_vectors,
)

# Use a non-interactive backend for testing to prevent plots from showing
matplotlib.use("Agg")


# sample tile_metrics.json
sample_metrics_data = {
    "488_561": {
        "Tile_X_0000_Y_0000_Z_0000": {"affine_transform": [[1, 0, 1.2], [0, 1, -0.8], [0, 0, 1]]},
        "Tile_X_0001_Y_0000_Z_0000": {"affine_transform": [[1, 0, 2.4], [0, 1, -0.6], [0, 0, 1]]},
        "Tile_X_0000_Y_0001_Z_0000": {"affine_transform": [[1, 0, 1.5], [0, 1, 0.4], [0, 0, 1]]},
    },
    "561_638": {
        "Tile_X_0000_Y_0000_Z_0000": {"affine_transform": [[1, 0, -2.2], [0, 1, 1.0], [0, 0, 1]]},
        "Tile_X_0001_Y_0000_Z_0000": {"affine_transform": [[1, 0, -2.4], [0, 1, 1.3], [0, 0, 1]]},
        "Tile_X_0000_Y_0001_Z_0000": {"affine_transform": [[1, 0, -1.9], [0, 1, 0.6], [0, 0, 1]]},
    },
    "638_750": {
        "Tile_X_0000_Y_0000_Z_0000": {"affine_transform": [[1, 0, 0.4], [0, 1, 0.2], [0, 0, 1]]},
        "Tile_X_0001_Y_0000_Z_0000": {"affine_transform": [[1, 0, 0.3], [0, 1, 0.1], [0, 0, 1]]},
        "Tile_X_0000_Y_0001_Z_0000": {"affine_transform": [[1, 0, 0.6], [0, 1, -0.2], [0, 0, 1]]},
    },
}


class TestCameraAlignment(unittest.TestCase):
    """Test class for camera alignment functions."""

    def test_extract_translation_stats(self):
        """Verify calculation of translation stats per pair and overall."""
        stats, dx_all, dy_all, d_all = _extract_translation_stats(sample_metrics_data)

        # Expected values calculated manually
        expected_stats = {
            "488_561": {
                "dx": np.array([1.2, 2.4, 1.5]),
                "dy": np.array([-0.8, -0.6, 0.4]),
                "d": np.hypot([1.2, 2.4, 1.5], [-0.8, -0.6, 0.4]),
            },
            "561_638": {
                "dx": np.array([-2.2, -2.4, -1.9]),
                "dy": np.array([1.0, 1.3, 0.6]),
                "d": np.hypot([-2.2, -2.4, -1.9], [1.0, 1.3, 0.6]),
            },
            "638_750": {
                "dx": np.array([0.4, 0.3, 0.6]),
                "dy": np.array([0.2, 0.1, -0.2]),
                "d": np.hypot([0.4, 0.3, 0.6], [0.2, 0.1, -0.2]),
            },
        }

        expected_dx_all = np.array([1.2, 2.4, 1.5, -2.2, -2.4, -1.9, 0.4, 0.3, 0.6])
        expected_dy_all = np.array([-0.8, -0.6, 0.4, 1.0, 1.3, 0.6, 0.2, 0.1, -0.2])
        expected_d_all = np.concatenate([
            expected_stats["488_561"]["d"],
            expected_stats["561_638"]["d"],
            expected_stats["638_750"]["d"],
        ])

        # Assertions
        self.assertEqual(stats.keys(), expected_stats.keys())

        for pair in expected_stats:
            self.assertEqual(stats[pair].keys(), expected_stats[pair].keys())
            assert_allclose(stats[pair]["dx"], expected_stats[pair]["dx"], rtol=1e-5, atol=1e-8)
            assert_allclose(stats[pair]["dy"], expected_stats[pair]["dy"], rtol=1e-5, atol=1e-8)
            assert_allclose(stats[pair]["d"], expected_stats[pair]["d"], rtol=1e-5, atol=1e-8)

        assert_allclose(dx_all, expected_dx_all, rtol=1e-5, atol=1e-8)
        assert_allclose(dy_all, expected_dy_all, rtol=1e-5, atol=1e-8)
        assert_allclose(d_all, expected_d_all, rtol=1e-5, atol=1e-8)

    def test_plot_translation_distributions_runs(self):
        """Check if plot_translation_distributions runs and returns expected types."""
        try:
            fig, axes = plot_translation_distributions(sample_metrics_data)
            self.assertIsInstance(fig, Figure)
            self.assertIsInstance(axes, np.ndarray)
            # Check that the elements in the axes array are Axes objects
            self.assertTrue(all(isinstance(ax, Axes) for ax in axes.flat))
            # Check if the figure has axes
            self.assertGreater(len(fig.axes), 0)
        finally:
            # Close the plot to free memory
            plt.close(fig if "fig" in locals() else "all")

    def test_plot_translation_vectors_runs(self):
        """Check if plot_translation_vectors runs and returns expected types."""
        try:
            fig, ax = plot_translation_vectors(sample_metrics_data)
            self.assertIsInstance(fig, Figure)
            self.assertIsInstance(ax, Axes)
            # Check if the figure has axes (should be just the one)
            self.assertEqual(len(fig.axes), 1)
            self.assertIs(fig.axes[0], ax)
        finally:
            # Close the plot to free memory
            plt.close(fig if "fig" in locals() else "all")

    def test_load_tile_metrics(self):
        """Test loading tile metrics from JSON file."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_metrics_data, f)
            temp_path = Path(f.name)
        
        try:
            # Test loading
            loaded_data = load_tile_metrics(temp_path)
            self.assertEqual(loaded_data, sample_metrics_data)
        finally:
            # Clean up
            temp_path.unlink()


if __name__ == '__main__':
    unittest.main()
