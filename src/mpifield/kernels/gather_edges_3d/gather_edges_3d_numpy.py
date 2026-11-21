"""The NumPy version of the ``gather_edges_3d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_edges_3d_pyccel.py``, vectorized over the
markers (`mpifield.kernels.whitney_cell.gather_edges`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.whitney_cell import gather_edges


def gather_edges_3d(
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
) -> None:
    """Write the field of ``edges`` (nx, ny, nz, 3) at ``positions`` into ``result`` (3, N)."""
    gather_edges(
        positions,
        edges,
        result,
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
