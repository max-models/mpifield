"""The NumPy version of the ``deposit_paths_spline_1d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``deposit_paths_spline_1d_pyccel.py``, vectorized over
the paths (`mpifield.kernels.spline_edges`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_edges import deposit_paths_spline


def deposit_paths_spline_1d(
    start: np.ndarray,
    end: np.ndarray,
    charges: np.ndarray,
    edges: np.ndarray,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
    degree: int,
) -> None:
    """Add the current of ``charges`` along the paths from ``start`` to ``end`` to ``edges``."""
    deposit_paths_spline(
        start,
        end,
        charges,
        edges,
        np.array([lower_x]),
        np.array([cell_size_x]),
        np.array([num_x]),
        np.array([periodic_x]),
        degree,
    )
