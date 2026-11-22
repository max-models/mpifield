"""The NumPy version of the ``gather_edges_spline_3d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_edges_spline_3d_pyccel.py``, vectorized over
the markers (`mpifield.kernels.spline_edges`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_edges import gather_edges_spline


def gather_edges_spline_3d(
    positions: np.ndarray,
    edges: np.ndarray,
    result: np.ndarray,
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
    degree: int,
) -> None:
    """Write the field of ``edges`` at ``positions`` into ``result``."""
    gather_edges_spline(
        positions,
        edges,
        result,
        np.array([lower_x, lower_y, lower_z]),
        np.array([cell_size_x, cell_size_y, cell_size_z]),
        np.array([num_x, num_y, num_z]),
        np.array([periodic_x, periodic_y, periodic_z]).astype(bool),
        degree,
    )
