"""The NumPy version of the ``gather_linear_1d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_linear_1d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.linear_cell import linear_cell


def gather_linear_1d(
    positions: np.ndarray,
    field: np.ndarray,
    result: np.ndarray,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> None:
    """Interpolate ``field`` (nx, components) at the markers into ``result`` (components, N)."""
    left, right, fx = linear_cell(
        (positions[0] - lower_x) / cell_size_x, num_x, bool(periodic_x)
    )
    result[...] = (1.0 - fx) * field[left].T + fx * field[right].T
