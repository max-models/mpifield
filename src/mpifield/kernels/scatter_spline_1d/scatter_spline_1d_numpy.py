"""The NumPy version of the ``scatter_spline_1d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``scatter_spline_1d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_cell import spline_cell


def scatter_spline_1d(
    positions: np.ndarray,
    values: np.ndarray,
    field: np.ndarray,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
    degree: int,
) -> None:
    """Deposit ``values`` (components, N) onto ``field`` (nx, components)."""
    # zero-weight markers (dead ones left in storage) deposit nothing, whatever
    # their position, as in the compiled version
    deposit = np.any(values != 0.0, axis=0)
    if not deposit.all():
        positions, values = positions[:, deposit], values[:, deposit]
    nodes_x, weights_x = spline_cell(
        (positions[0] - lower_x) / cell_size_x, num_x, bool(periodic_x), degree
    )
    for i, wx in zip(nodes_x, weights_x, strict=True):
        np.add.at(field, i, (wx * values).T)
