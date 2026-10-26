"""The NumPy version of the ``gather_linear_2d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_linear_2d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.linear_cell import linear_cell


def gather_linear_2d(
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
    """Interpolate a 2-D grid field (nx, ny, components) into ``result`` (components, N)."""
    i_left, i_right, fx = linear_cell(
        (positions[0] - lower_x) / cell_size_x, num_x, bool(periodic_x)
    )
    j_left, j_right, fy = linear_cell(
        (positions[1] - lower_y) / cell_size_y, num_y, bool(periodic_y)
    )
    result[...] = (
        (1.0 - fx) * (1.0 - fy) * field[i_left, j_left].T
        + fx * (1.0 - fy) * field[i_right, j_left].T
        + (1.0 - fx) * fy * field[i_left, j_right].T
        + fx * fy * field[i_right, j_right].T
    )
