"""The NumPy version of the ``scatter_quadratic_1d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``scatter_quadratic_1d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.quadratic_cell import quadratic_cell


def scatter_quadratic_1d(
    positions: np.ndarray,
    values: np.ndarray,
    field: np.ndarray,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> None:
    """Deposit ``values`` (components, N) onto ``field`` (nx, components)."""
    # zero-weight markers (dead ones left in storage) deposit nothing, whatever
    # their position, as in the compiled version
    deposit = np.any(values != 0.0, axis=0)
    if not deposit.all():
        positions, values = positions[:, deposit], values[:, deposit]
    nodes_x, weights_x = quadratic_cell(
        (positions[0] - lower_x) / cell_size_x, num_x, bool(periodic_x)
    )
    for i, wx in zip(nodes_x, weights_x, strict=True):
        np.add.at(field, i, (wx * values).T)
