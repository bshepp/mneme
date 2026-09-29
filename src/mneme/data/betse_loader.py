"""BETSE data loader for importing bioelectric simulation output into Mneme.

Reads CSV exports from the BioElectric Tissue Simulation Engine (BETSE)
and converts them into Mneme Field objects suitable for the analysis pipeline.

BETSE outputs irregular cell-center data (x, y, Vmem) which this loader
interpolates onto a regular grid for field reconstruction and topology analysis.

References
----------
- BETSE: https://github.com/betsee/betse
- Pietak & Levin (2016). Frontiers in Bioengineering and Biotechnology, 4:55.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
import warnings
import re
import logging

from ..types import Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Default grid resolution for interpolating irregular BETSE cell data.
DEFAULT_GRID_RESOLUTION: Tuple[int, int] = (64, 64)

#: Column names expected in BETSE 2D Vmem CSV exports.
VMEM_2D_COLUMNS = ("x [um]", "y [um]", "Vmem [mV]")

#: Default interpolation. Linear stays within the range of the cell data;
#: cubic overshoots and creates voltages that are not in the simulation.
DEFAULT_INTERPOLATION = "linear"

#: The frame index is the trailing integer of the file stem ("Vmem2D_17").
#: Anchoring at the end matters: an unanchored search matches the "2" in
#: "Vmem2D" for every file.
_FRAME_INDEX_RE = re.compile(r"_(\d+)$")


def _frame_index(path: Path) -> int:
    """Return the time index encoded in a ``Vmem2D_<n>.csv`` file name."""
    match = _FRAME_INDEX_RE.search(path.stem)
    if match is None:
        raise ValueError(
            f"Cannot read a frame index from {path.name!r}; "
            f"expected a name like 'Vmem2D_17.csv'"
        )
    return int(match.group(1))


def _sorted_frame_files(vmem_dir: Path) -> List[Path]:
    """List ``Vmem2D_*.csv`` files in numeric time order."""
    csv_files = sorted(vmem_dir.glob("Vmem2D_*.csv"), key=_frame_index)
    if not csv_files:
        raise FileNotFoundError(f"No Vmem2D_*.csv files found in {vmem_dir}")
    return csv_files


def load_betse_vmem_csv(
    csv_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single BETSE Vmem2D CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to a Vmem2D_*.csv file exported by BETSE.

    Returns
    -------
    x : np.ndarray
        Cell x-coordinates in micrometres, shape (n_cells,).
    y : np.ndarray
        Cell y-coordinates in micrometres, shape (n_cells,).
    vmem : np.ndarray
        Transmembrane voltage in millivolts, shape (n_cells,).
    """
    csv_path = Path(csv_path)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"Expected CSV with at least 3 columns (x, y, Vmem), "
            f"got shape {data.shape} from {csv_path}"
        )
    return data[:, 0], data[:, 1], data[:, 2]


def interpolate_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    resolution: Tuple[int, int] = DEFAULT_GRID_RESOLUTION,
    method: str = DEFAULT_INTERPOLATION,
    padding: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate scattered cell data onto a regular grid.

    Parameters
    ----------
    x, y : np.ndarray
        Irregular cell positions, each shape (n_cells,).
    values : np.ndarray
        Scalar values at cell positions, shape (n_cells,).
    resolution : Tuple[int, int]
        Target grid resolution (rows, cols).
    method : str
        Interpolation method ('linear', 'cubic', 'nearest').
    padding : float
        Fractional padding around data extent (0.05 = 5%).

    Returns
    -------
    grid : np.ndarray
        Interpolated field, shape *resolution*.
    grid_x : np.ndarray
        1-D array of x-coordinates for the regular grid.
    grid_y : np.ndarray
        1-D array of y-coordinates for the regular grid.
    """
    # Compute data extent with padding
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    pad_x = x_range * padding
    pad_y = y_range * padding

    grid_x = np.linspace(x.min() - pad_x, x.max() + pad_x, resolution[1])
    grid_y = np.linspace(y.min() - pad_y, y.max() + pad_y, resolution[0])
    gx, gy = np.meshgrid(grid_x, grid_y)

    points = np.column_stack([x, y])
    grid = griddata(points, values, (gx, gy), method=method)

    # Fill NaN regions (outside convex hull) with nearest-neighbour values
    if np.any(np.isnan(grid)):
        grid_nn = griddata(points, values, (gx, gy), method="nearest")
        grid = np.where(np.isnan(grid), grid_nn, grid)

    return grid, grid_x, grid_y


def load_betse_cells(
    vmem_dir: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Load a BETSE Vmem2D time series at the cells, without interpolation.

    Prefer this over :func:`load_betse_timeseries` for any analysis that
    does not need a regular grid: it contains only values the simulation
    produced.

    Parameters
    ----------
    vmem_dir : str or Path
        Directory containing Vmem2D_0.csv, Vmem2D_1.csv, ... files.

    Returns
    -------
    vmem : np.ndarray
        Shape (n_timesteps, n_cells), in mV, in time order.
    x, y : np.ndarray
        Cell positions in micrometres, shape (n_cells,), from the first frame.
    frame_indices : List[int]
        Time index of each row of ``vmem``.
    """
    vmem_dir = Path(vmem_dir)
    csv_files = _sorted_frame_files(vmem_dir)
    x0, y0, first = load_betse_vmem_csv(csv_files[0])
    rows = [first]
    for csv_path in csv_files[1:]:
        _, _, vmem = load_betse_vmem_csv(csv_path)
        if vmem.shape != first.shape:
            raise ValueError(
                f"{csv_path.name} has {vmem.shape[0]} cells; "
                f"{csv_files[0].name} has {first.shape[0]}"
            )
        rows.append(vmem)
    return np.stack(rows, axis=0), x0, y0, [_frame_index(p) for p in csv_files]


def load_betse_timeseries(
    vmem_dir: Union[str, Path],
    resolution: Tuple[int, int] = DEFAULT_GRID_RESOLUTION,
    method: str = DEFAULT_INTERPOLATION,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a BETSE Vmem2D time series from a directory of CSV files.

    Reads all ``Vmem2D_*.csv`` files, interpolates each onto a common
    regular grid, and stacks them into a 3-D array (time, rows, cols).

    Parameters
    ----------
    vmem_dir : str or Path
        Directory containing Vmem2D_0.csv, Vmem2D_1.csv, ... files.
    resolution : Tuple[int, int]
        Target grid resolution per frame.
    method : str
        Interpolation method passed to :func:`interpolate_to_grid`.

    Returns
    -------
    field_sequence : np.ndarray
        Shape (n_timesteps, *resolution) array of Vmem fields in mV.
    metadata : Dict[str, Any]
        Dictionary with keys:
        - ``n_cells``: number of cells in the simulation
        - ``n_timesteps``: number of temporal frames loaded
        - ``frame_indices``: time index of each frame, ascending
        - ``inside_hull``: boolean (rows, cols) mask, True where the grid
          point lies inside the convex hull of the cells. Values outside
          it are nearest-neighbour fill, not interpolation.
        - ``grid_x``, ``grid_y``: 1-D coordinate arrays for the regular grid
        - ``x_bounds``, ``y_bounds``: spatial extent in micrometres
        - ``source_dir``: path to source directory
    """
    vmem_dir = Path(vmem_dir)
    csv_files = _sorted_frame_files(vmem_dir)

    logger.info(
        "Loading %d BETSE Vmem frames from %s at resolution %s",
        len(csv_files), vmem_dir, resolution,
    )

    # First pass: determine shared coordinate bounds from all frames
    all_x, all_y = [], []
    frame_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for csv_path in csv_files:
        x, y, vmem = load_betse_vmem_csv(csv_path)
        frame_data.append((x, y, vmem))
        all_x.append(x)
        all_y.append(y)

    # Use union of all coordinate extents for a consistent grid
    all_x_arr = np.concatenate(all_x)
    all_y_arr = np.concatenate(all_y)
    padding = 0.05
    x_min = all_x_arr.min() - (all_x_arr.max() - all_x_arr.min()) * padding
    x_max = all_x_arr.max() + (all_x_arr.max() - all_x_arr.min()) * padding
    y_min = all_y_arr.min() - (all_y_arr.max() - all_y_arr.min()) * padding
    y_max = all_y_arr.max() + (all_y_arr.max() - all_y_arr.min()) * padding

    grid_x = np.linspace(x_min, x_max, resolution[1])
    grid_y = np.linspace(y_min, y_max, resolution[0])
    gx, gy = np.meshgrid(grid_x, grid_y)

    # Second pass: interpolate each frame onto the shared grid
    frames = []
    for i, (x, y, vmem) in enumerate(frame_data):
        points = np.column_stack([x, y])
        grid = griddata(points, vmem, (gx, gy), method=method)
        if np.any(np.isnan(grid)):
            grid_nn = griddata(points, vmem, (gx, gy), method="nearest")
            grid = np.where(np.isnan(grid), grid_nn, grid)
        frames.append(grid)

    field_sequence = np.stack(frames, axis=0)

    x0, y0, _ = frame_data[0]
    hull = Delaunay(np.column_stack([x0, y0]))
    inside_hull = (
        hull.find_simplex(np.column_stack([gx.ravel(), gy.ravel()])) >= 0
    ).reshape(gx.shape)

    metadata = {
        "n_cells": len(frame_data[0][0]),
        "n_timesteps": len(frames),
        "frame_indices": [_frame_index(p) for p in csv_files],
        "inside_hull": inside_hull,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "x_bounds": (float(x_min), float(x_max)),
        "y_bounds": (float(y_min), float(y_max)),
        "resolution": resolution,
        "interpolation_method": method,
        "source_dir": str(vmem_dir),
        "source": "BETSE",
        "units": {"spatial": "um", "voltage": "mV"},
    }

    logger.info(
        "Loaded BETSE field sequence: shape=%s, Vmem range=[%.2f, %.2f] mV",
        field_sequence.shape, field_sequence.min(), field_sequence.max(),
    )

    return field_sequence, metadata


def load_betse_exported_data(
    csv_path: Union[str, Path],
) -> Tuple[np.ndarray, List[str]]:
    """Load a BETSE ExportedData.csv single-cell time series.

    Parameters
    ----------
    csv_path : str or Path
        Path to ExportedData.csv.

    Returns
    -------
    data : np.ndarray
        Shape (n_timesteps, n_columns) array.
    columns : List[str]
        Column names from the CSV header.
    """
    csv_path = Path(csv_path)
    with open(csv_path, "r") as f:
        header = f.readline().strip()
    columns = [c.strip() for c in header.split(",")]
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data, columns


def betse_to_field(
    vmem_dir: Union[str, Path],
    resolution: Tuple[int, int] = DEFAULT_GRID_RESOLUTION,
    method: str = DEFAULT_INTERPOLATION,
) -> Field:
    """Load BETSE data and return a Mneme Field object.

    This is the main entry point for feeding BETSE simulation output
    into the Mneme analysis pipeline.

    Parameters
    ----------
    vmem_dir : str or Path
        Directory containing Vmem2D_*.csv files.
    resolution : Tuple[int, int]
        Grid resolution for interpolation.
    method : str
        Interpolation method ('linear', 'cubic', 'nearest').

    Returns
    -------
    field : Field
        Mneme Field with shape (n_timesteps, rows, cols) if multiple
        frames, or (rows, cols) for a single frame.
    """
    field_sequence, metadata = load_betse_timeseries(
        vmem_dir, resolution=resolution, method=method
    )

    # Build 2-D coordinate array for the regular grid
    grid_x = metadata["grid_x"]
    grid_y = metadata["grid_y"]
    gx, gy = np.meshgrid(grid_x, grid_y)
    coordinates = np.column_stack([gx.ravel(), gy.ravel()])

    # Squeeze out time dimension if only one frame
    data = field_sequence if field_sequence.shape[0] > 1 else field_sequence[0]

    return Field(
        data=data,
        coordinates=coordinates,
        resolution=resolution,
        bounds=(
            (metadata["x_bounds"][0], metadata["x_bounds"][1]),
            (metadata["y_bounds"][0], metadata["y_bounds"][1]),
        ),
        metadata=metadata,
    )
