"""The NumPy version of the ``gather_quadratic_2d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_quadratic_2d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.quadratic_cell import quadratic_cell


def gather_quadratic_2d(
    positions: np.ndarray,
    field: np.ndarray,
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
    """Interpolate ``field`` (nx, ny, components) at the markers into ``result`` (components, N)."""
    nodes_x, weights_x = quadratic_cell(
        (positions[0] - lower_x) / cell_size_x, num_x, bool(periodic_x)
    )
    nodes_y, weights_y = quadratic_cell(
        (positions[1] - lower_y) / cell_size_y, num_y, bool(periodic_y)
    )
    total = np.zeros_like(result)
    for i, wx in zip(nodes_x, weights_x, strict=True):
        for j, wy in zip(nodes_y, weights_y, strict=True):
            total += wx * wy * field[i, j].T
    result[...] = total
