"""The NumPy version of the ``scatter_linear_2d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``scatter_linear_2d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.linear_cell import linear_cell


def scatter_linear_2d(
    positions: np.ndarray,
    values: np.ndarray,
    field: np.ndarray,
    lower_x: float,
    lower_y: float,
    cell_size_x: float,
    cell_size_y: float,
    num_x: int,
    num_y: int,
    periodic_x: int,
    periodic_y: int,
) -> None:
    """Deposit ``values`` (components, N) onto ``field`` (nx, ny, components)."""
    # zero-weight markers (dead ones left in storage) deposit nothing, whatever
    # their position, as in the compiled version
    deposit = np.any(values != 0.0, axis=0)
    if not deposit.all():
        positions, values = positions[:, deposit], values[:, deposit]
    i_left, i_right, fx = linear_cell(
        (positions[0] - lower_x) / cell_size_x, num_x, bool(periodic_x)
    )
    j_left, j_right, fy = linear_cell(
        (positions[1] - lower_y) / cell_size_y, num_y, bool(periodic_y)
    )
    flat = field.reshape(num_x * num_y, field.shape[-1])  # a view
    for i, j, weight in (
        (i_left, j_left, (1.0 - fx) * (1.0 - fy)),
        (i_right, j_left, fx * (1.0 - fy)),
        (i_left, j_right, (1.0 - fx) * fy),
        (i_right, j_right, fx * fy),
    ):
        np.add.at(flat, i * num_y + j, (weight * values).T)
