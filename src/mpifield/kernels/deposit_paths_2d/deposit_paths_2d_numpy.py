"""The NumPy version of the ``deposit_paths_2d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``deposit_paths_2d_pyccel.py``, vectorized over the
pieces of the paths (`mpifield.kernels.whitney_cell.deposit_paths`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.whitney_cell import deposit_paths


def deposit_paths_2d(
    start: np.ndarray,
    end: np.ndarray,
    charges: np.ndarray,
    edges: np.ndarray,
    lower_x: float,
    lower_y: float,
    cell_size_x: float,
    cell_size_y: float,
    num_x: int,
    num_y: int,
    periodic_x: int,
    periodic_y: int,
) -> None:
    """Add the current of the paths from ``start`` to ``end`` (2, N) to ``edges`` (nx, ny, 2)."""
    deposit_paths(
        start,
        end,
        charges,
        edges,
        (
            lower_x,
            lower_y,
            cell_size_x,
            cell_size_y,
            num_x,
            num_y,
            periodic_x,
            periodic_y,
        ),
    )
