"""The NumPy version of the ``scatter_linear_1d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``scatter_linear_1d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.linear_cell import linear_cell


def scatter_linear_1d(
    positions: np.ndarray,
    values: np.ndarray,
    field: np.ndarray,
    lower_x: float,
    cell_size: float,
    num_gridpoints: int,
    periodic_x: int,
) -> None:
    """Deposit scalar marker values onto ``field`` with linear weights."""
    # zero-weight markers (dead ones left in storage) deposit nothing, whatever
    # their position, as in the compiled version
    deposit = values != 0.0
    if not deposit.all():
        positions, values = positions[deposit], values[deposit]
    left, right, fraction = linear_cell(
        (positions - lower_x) / cell_size, num_gridpoints, bool(periodic_x)
    )
    # bincount sums repeated cells in compiled code, much faster than np.add.at
    field += np.bincount(
        left, weights=(1.0 - fraction) * values, minlength=field.shape[0]
    )
    field += np.bincount(right, weights=fraction * values, minlength=field.shape[0])
