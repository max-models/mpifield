"""The NumPy version of the ``gather_forms_spline_3d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_forms_spline_3d_pyccel.py``, vectorized over
the markers (`mpifield.kernels.spline_forms`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_forms import gather_forms_spline


def gather_forms_spline_3d(
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
    degree: int,
    form: int,
) -> None:
    """The gather of a k-form (`mpifield.kernels.spline_forms.gather_forms_spline`)."""
    gather_forms_spline(
        positions,
        field,
        result,
        np.array([lower_x, lower_y, lower_z]),
        np.array([cell_size_x, cell_size_y, cell_size_z]),
        np.array([num_x, num_y, num_z]),
        np.array([periodic_x, periodic_y, periodic_z]),
        degree,
        form,
    )
