"""The NumPy version of the ``deposit_paths_1d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``deposit_paths_1d_pyccel.py``, vectorized over the
pieces of the paths (`mpifield.kernels.whitney_cell.deposit_paths`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.whitney_cell import deposit_paths


def deposit_paths_1d(
    start: np.ndarray,
    end: np.ndarray,
    charges: np.ndarray,
    edges: np.ndarray,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> None:
    """Add the current of the paths from ``start`` to ``end`` (1, N) to ``edges`` (nx, 1)."""
    deposit_paths(
        start,
        end,
        charges,
        edges,
        (
            lower_x,
            cell_size_x,
            num_x,
            periodic_x,
        ),
    )
