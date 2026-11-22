"""The NumPy version of the ``gather_edges_spline_1d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_edges_spline_1d_pyccel.py``, vectorized over
the markers (`mpifield.kernels.spline_edges`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_edges import gather_edges_spline


def gather_edges_spline_1d(
    positions: np.ndarray,
    edges: np.ndarray,
    result: np.ndarray,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
    degree: int,
) -> None:
    """Write the field of ``edges`` at ``positions`` into ``result``."""
    gather_edges_spline(
        positions,
        edges,
        result,
        np.array([lower_x]),
        np.array([cell_size_x]),
        np.array([num_x]),
        np.array([periodic_x]),
        degree,
    )
