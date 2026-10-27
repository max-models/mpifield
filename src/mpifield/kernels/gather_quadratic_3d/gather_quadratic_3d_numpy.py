"""The NumPy version of the ``gather_quadratic_3d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_quadratic_3d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.quadratic_cell import quadratic_cell


def gather_quadratic_3d(
    positions: np.ndarray,
    field: np.ndarray,
    result: np.ndarray,
    lower_x: float,
    lower_y: float,
    lower_z: float,
    cell_size_x: float,
    cell_size_y: float,
    cell_size_z: float,
    num_x: int,
    num_y: int,
    num_z: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
) -> None:
    """Interpolate ``field`` (nx, ny, nz, components) at the markers into ``result`` (components, N)."""
    nodes_x, weights_x = quadratic_cell(
        (positions[0] - lower_x) / cell_size_x, num_x, bool(periodic_x)
    )
    nodes_y, weights_y = quadratic_cell(
        (positions[1] - lower_y) / cell_size_y, num_y, bool(periodic_y)
    )
    nodes_z, weights_z = quadratic_cell(
        (positions[2] - lower_z) / cell_size_z, num_z, bool(periodic_z)
    )
    total = np.zeros_like(result)
    for i, wx in zip(nodes_x, weights_x, strict=True):
        for j, wy in zip(nodes_y, weights_y, strict=True):
            for k, wz in zip(nodes_z, weights_z, strict=True):
                total += wx * wy * wz * field[i, j, k].T
    result[...] = total
