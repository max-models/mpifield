"""The NumPy version of the ``deposit_paths_spline_2d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``deposit_paths_spline_2d_pyccel.py``, vectorized over
the paths (`mpifield.kernels.spline_edges`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_edges import deposit_paths_spline


def deposit_paths_spline_2d(
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
    degree: int,
) -> None:
    """Add the current of ``charges`` along the paths from ``start`` to ``end`` to ``edges``."""
    deposit_paths_spline(
        start,
        end,
        charges,
        edges,
        np.array([lower_x, lower_y]),
        np.array([cell_size_x, cell_size_y]),
        np.array([num_x, num_y]),
        np.array([periodic_x, periodic_y]),
        degree,
    )
