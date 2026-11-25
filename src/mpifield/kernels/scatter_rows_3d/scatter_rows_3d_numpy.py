"""The NumPy version of the ``scatter_rows_3d`` kernel: ``scatter_linear_3d``'s on the rows."""

from __future__ import annotations

import numpy as np

from mpifield.kernels.scatter_linear_3d.scatter_linear_3d_numpy import scatter_linear_3d


def scatter_rows_3d(
    px, py, pz, values, field,
    lower_x, lower_y, lower_z, cell_size_x, cell_size_y, cell_size_z,
    num_x, num_y, num_z, periodic_x, periodic_y, periodic_z,
) -> None:  # fmt: skip
    """Add ``values`` at the positions to ``field`` (see the Pyccel version)."""
    scatter_linear_3d(
        np.stack([px, py, pz]),
        values[None, :],
        field.reshape(-1, 1),
        lower_x, lower_y, lower_z, cell_size_x, cell_size_y, cell_size_z,
        num_x, num_y, num_z, periodic_x, periodic_y, periodic_z,
    )  # fmt: skip
