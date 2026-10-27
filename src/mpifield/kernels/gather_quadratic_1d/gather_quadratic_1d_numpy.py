"""The NumPy version of the ``gather_quadratic_1d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_quadratic_1d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.quadratic_cell import quadratic_cell


def gather_quadratic_1d(
    positions: np.ndarray,
    field: np.ndarray,
    result: np.ndarray,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> None:
    """Interpolate ``field`` (nx, components) at the markers into ``result`` (components, N)."""
    nodes_x, weights_x = quadratic_cell(
        (positions[0] - lower_x) / cell_size_x, num_x, bool(periodic_x)
    )
    total = np.zeros_like(result)
    for i, wx in zip(nodes_x, weights_x, strict=True):
        total += wx * field[i].T
    result[...] = total
