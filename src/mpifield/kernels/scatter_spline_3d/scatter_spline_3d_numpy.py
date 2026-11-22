"""The NumPy version of the ``scatter_spline_3d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``scatter_spline_3d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_cell import spline_cell


def scatter_spline_3d(
    positions: np.ndarray,
    values: np.ndarray,
    field: np.ndarray,
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
    degree: int,
) -> None:
    """Deposit ``values`` (components, N) onto ``field`` (nx, ny, nz, components)."""
    # zero-weight markers (dead ones left in storage) deposit nothing, whatever
    # their position, as in the compiled version
    deposit = np.any(values != 0.0, axis=0)
    if not deposit.all():
        positions, values = positions[:, deposit], values[:, deposit]
    nodes_x, weights_x = spline_cell(
        (positions[0] - lower_x) / cell_size_x, num_x, int(periodic_x), degree
    )
    nodes_y, weights_y = spline_cell(
        (positions[1] - lower_y) / cell_size_y, num_y, int(periodic_y), degree
    )
    nodes_z, weights_z = spline_cell(
        (positions[2] - lower_z) / cell_size_z, num_z, int(periodic_z), degree
    )
    flat = field.reshape(-1, field.shape[-1])  # a view
    for i, wx in zip(nodes_x, weights_x, strict=True):
        for j, wy in zip(nodes_y, weights_y, strict=True):
            for k, wz in zip(nodes_z, weights_z, strict=True):
                np.add.at(
                    flat,
                    (i * field.shape[1] + j) * field.shape[2] + k,
                    (wx * wy * wz * values).T,
                )
