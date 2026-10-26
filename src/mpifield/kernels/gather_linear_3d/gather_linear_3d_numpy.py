"""The NumPy version of the ``gather_linear_3d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_linear_3d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.linear_cell import linear_cell


def gather_linear_3d(
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
    """Interpolate a 3D grid field at the markers into ``result`` (components, N).

    ``field`` is the grid ``(nx, ny, nz, components)`` reshaped to
    ``(nx * ny * nz, components)``: cell ``(i, j, k)`` is row ``(i * ny + j) * nz + k``.
    """
    cells = [
        linear_cell((positions[d] - lower) / size, num, bool(periodic))
        for d, (lower, size, num, periodic) in enumerate(
            (
                (lower_x, cell_size_x, num_x, periodic_x),
                (lower_y, cell_size_y, num_y, periodic_y),
                (lower_z, cell_size_z, num_z, periodic_z),
            )
        )
    ]
    value = np.zeros(result.shape, dtype=np.result_type(field, float))
    for a in range(2):
        i, wx = cells[0][a], cells[0][2] if a else 1.0 - cells[0][2]
        for b in range(2):
            j, wy = cells[1][b], cells[1][2] if b else 1.0 - cells[1][2]
            for c in range(2):
                k, wz = cells[2][c], cells[2][2] if c else 1.0 - cells[2][2]
                value += (wx * wy * wz) * field[(i * num_y + j) * num_z + k].T
    result[...] = value
