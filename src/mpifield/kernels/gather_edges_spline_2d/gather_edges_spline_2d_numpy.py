"""The NumPy version of the ``gather_edges_spline_2d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_edges_spline_2d_pyccel.py``, vectorized over
the markers (`mpifield.kernels.spline_edges`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_edges import gather_edges_spline


def gather_edges_spline_2d(
    positions: np.ndarray,
    edges: np.ndarray,
    result: np.ndarray,
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
    """Write the field of ``edges`` at ``positions`` into ``result``."""
    gather_edges_spline(
        positions,
        edges,
        result,
        np.array([lower_x, lower_y]),
        np.array([cell_size_x, cell_size_y]),
        np.array([num_x, num_y]),
        np.array([periodic_x, periodic_y]),
        degree,
    )
