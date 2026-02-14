"""Tests for mneme.data.betse_loader — BETSE data loading and interpolation."""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from mneme.data.betse_loader import (
    load_betse_vmem_csv,
    interpolate_to_grid,
    load_betse_timeseries,
    load_betse_exported_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_vmem_csv(path: Path, x, y, vmem):
    """Write a synthetic Vmem2D CSV file."""
    with open(path, "w") as f:
        f.write("x [um],y [um],Vmem [mV]\n")
        for xi, yi, vi in zip(x, y, vmem):
            f.write(f"{xi},{yi},{vi}\n")


def _make_circle_cells(n_cells: int = 50, radius: float = 100.0, seed: int = 42):
    """Generate synthetic cell positions in a disc."""
    rng = np.random.RandomState(seed)
    # Uniform sampling in a circle
    angles = rng.uniform(0, 2 * np.pi, n_cells)
    radii = radius * np.sqrt(rng.uniform(0, 1, n_cells))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y


# ---------------------------------------------------------------------------
# load_betse_vmem_csv
# ---------------------------------------------------------------------------

class TestLoadBetseVmemCsv:
    """Tests for loading a single Vmem2D CSV file."""

    def test_basic_load(self, tmp_path):
        x = np.array([0.0, 10.0, 20.0])
        y = np.array([0.0, 10.0, 20.0])
        vmem = np.array([-50.0, -60.0, -70.0])
        csv_path = tmp_path / "Vmem2D_0.csv"
        _write_vmem_csv(csv_path, x, y, vmem)

        xr, yr, vr = load_betse_vmem_csv(csv_path)

        np.testing.assert_array_almost_equal(xr, x)
        np.testing.assert_array_almost_equal(yr, y)
        np.testing.assert_array_almost_equal(vr, vmem)

    def test_returns_correct_shapes(self, tmp_path):
        n_cells = 25
        x, y = _make_circle_cells(n_cells)
        vmem = np.random.randn(n_cells) * 10 - 50
        csv_path = tmp_path / "Vmem2D_0.csv"
        _write_vmem_csv(csv_path, x, y, vmem)

        xr, yr, vr = load_betse_vmem_csv(csv_path)

        assert xr.shape == (n_cells,)
        assert yr.shape == (n_cells,)
        assert vr.shape == (n_cells,)

    def test_bad_csv_raises(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w") as f:
            f.write("a,b\n1,2\n3,4\n")
        with pytest.raises(ValueError, match="at least 3 columns"):
            load_betse_vmem_csv(csv_path)


# ---------------------------------------------------------------------------
# interpolate_to_grid
# ---------------------------------------------------------------------------

class TestInterpolateToGrid:
    """Tests for scattered-to-grid interpolation."""

    def test_output_shape(self):
        x, y = _make_circle_cells(100)
        vmem = np.sin(x / 50) * np.cos(y / 50)
        grid, gx, gy = interpolate_to_grid(x, y, vmem, resolution=(32, 32))

        assert grid.shape == (32, 32)
        assert gx.shape == (32,)
        assert gy.shape == (32,)

    def test_no_nans(self):
        """Interpolation should fill NaN regions with nearest neighbour."""
        x, y = _make_circle_cells(100)
        vmem = x + y  # Simple linear field
        grid, _, _ = interpolate_to_grid(x, y, vmem, resolution=(16, 16))

        assert not np.any(np.isnan(grid))

    def test_different_resolutions(self):
        x, y = _make_circle_cells(50)
        vmem = np.ones_like(x) * -50.0
        for res in [(8, 8), (16, 32), (64, 64)]:
            grid, gx, gy = interpolate_to_grid(x, y, vmem, resolution=res)
            assert grid.shape == res

    def test_linear_method(self):
        x, y = _make_circle_cells(80)
        vmem = x * 0.5  # Linear in x
        grid, gx, _ = interpolate_to_grid(x, y, vmem, resolution=(16, 16), method="linear")

        assert grid.shape == (16, 16)
        assert not np.any(np.isnan(grid))


# ---------------------------------------------------------------------------
# load_betse_timeseries
# ---------------------------------------------------------------------------

class TestLoadBetseTimeseries:
    """Tests for loading a directory of Vmem CSVs as a time series."""

    def test_loads_sequence(self, tmp_path):
        n_cells = 30
        n_frames = 5
        x, y = _make_circle_cells(n_cells)

        for t in range(n_frames):
            vmem = np.sin(x / 50 + t * 0.3) * 10 - 50
            _write_vmem_csv(tmp_path / f"Vmem2D_{t}.csv", x, y, vmem)

        field_seq, metadata = load_betse_timeseries(
            tmp_path, resolution=(16, 16)
        )

        assert field_seq.shape == (n_frames, 16, 16)
        assert metadata["n_timesteps"] == n_frames
        assert metadata["n_cells"] == n_cells
        assert not np.any(np.isnan(field_seq))

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No Vmem2D"):
            load_betse_timeseries(tmp_path)

    def test_metadata_keys(self, tmp_path):
        x, y = _make_circle_cells(20)
        vmem = np.zeros(20) - 50
        _write_vmem_csv(tmp_path / "Vmem2D_0.csv", x, y, vmem)

        _, metadata = load_betse_timeseries(tmp_path, resolution=(8, 8))

        assert "grid_x" in metadata
        assert "grid_y" in metadata
        assert "x_bounds" in metadata
        assert "y_bounds" in metadata
        assert metadata["source"] == "BETSE"


# ---------------------------------------------------------------------------
# load_betse_exported_data
# ---------------------------------------------------------------------------

class TestLoadBetseExportedData:
    """Tests for loading ExportedData.csv single-cell time series."""

    def test_basic_load(self, tmp_path):
        csv_path = tmp_path / "ExportedData.csv"
        with open(csv_path, "w") as f:
            f.write("Time [s],Vmem [mV],Ca [uM]\n")
            for t in range(10):
                f.write(f"{t * 0.1},{-50 + t * 0.5},{0.1 + t * 0.01}\n")

        data, columns = load_betse_exported_data(csv_path)

        assert data.shape == (10, 3)
        assert len(columns) == 3
        assert "Time [s]" in columns
        assert "Vmem [mV]" in columns

    def test_single_column(self, tmp_path):
        csv_path = tmp_path / "ExportedData.csv"
        with open(csv_path, "w") as f:
            f.write("Vmem [mV]\n")
            for v in [-50, -49, -48]:
                f.write(f"{v}\n")

        data, columns = load_betse_exported_data(csv_path)

        assert data.shape == (3,)
        assert len(columns) == 1
