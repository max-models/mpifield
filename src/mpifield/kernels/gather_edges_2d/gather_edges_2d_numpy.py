"""The NumPy version of the ``gather_edges_2d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_edges_2d_pyccel.py``, vectorized over the
markers (`mpifield.kernels.whitney_cell.gather_edges`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.whitney_cell import gather_edges


def gather_edges_2d(
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
) -> None:
    """Write the field of ``edges`` (nx, ny, 2) at ``positions`` into ``result`` (2, N)."""
    gather_edges(
        positions,
        edges,
        result,
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
