"""The NumPy version of the ``gather_edges_1d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_edges_1d_pyccel.py``, vectorized over the
markers (`mpifield.kernels.whitney_cell.gather_edges`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.whitney_cell import gather_edges


def gather_edges_1d(
    positions: np.ndarray,
    edges: np.ndarray,
    result: np.ndarray,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> None:
    """Write the field of ``edges`` (nx, 1) at ``positions`` into ``result`` (1, N)."""
    gather_edges(
        positions,
        edges,
        result,
        (
            lower_x,
            cell_size_x,
            num_x,
            periodic_x,
        ),
    )
