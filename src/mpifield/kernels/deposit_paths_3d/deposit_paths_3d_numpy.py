"""The NumPy version of the ``deposit_paths_3d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``deposit_paths_3d_pyccel.py``, vectorized over the
pieces of the paths (`mpifield.kernels.whitney_cell.deposit_paths`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.whitney_cell import deposit_paths


def deposit_paths_3d(
    start: np.ndarray,
    end: np.ndarray,
    charges: np.ndarray,
    edges: np.ndarray,
    lower_x: float,
    lower_y: float,
    lower_z: float,
    cell_size_x: float,
    cell_size_y: float,
    cell_size_z: float,
    num_x: int,
    num_y: int,
    num_z: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
) -> None:
    """Add the current of the paths from ``start`` to ``end`` (3, N) to ``edges`` (nx, ny, nz, 3)."""
    deposit_paths(
        start,
        end,
        charges,
        edges,
        (
            lower_x,
            lower_y,
            lower_z,
            cell_size_x,
            cell_size_y,
            cell_size_z,
            num_x,
            num_y,
            num_z,
            periodic_x,
            periodic_y,
            periodic_z,
        ),
    )
