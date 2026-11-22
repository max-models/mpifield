"""The NumPy version of the ``gather_spline_2d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_spline_2d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_cell import spline_cell


def gather_spline_2d(
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
    degree: int,
) -> None:
    """Interpolate ``field`` (nx, ny, components) at the markers into ``result`` (components, N)."""
    nodes_x, weights_x = spline_cell(
        (positions[0] - lower_x) / cell_size_x, num_x, int(periodic_x), degree
    )
    nodes_y, weights_y = spline_cell(
        (positions[1] - lower_y) / cell_size_y, num_y, int(periodic_y), degree
    )
    total = np.zeros_like(result)
    for i, wx in zip(nodes_x, weights_x, strict=True):
        for j, wy in zip(nodes_y, weights_y, strict=True):
            total += wx * wy * field[i, j].T
    result[...] = total
