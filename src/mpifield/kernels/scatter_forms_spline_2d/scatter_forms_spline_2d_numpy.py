"""The NumPy version of the ``scatter_forms_spline_2d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``scatter_forms_spline_2d_pyccel.py``, vectorized over
the markers (`mpifield.kernels.spline_forms`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_forms import scatter_forms_spline


def scatter_forms_spline_2d(
    positions: np.ndarray,
    values: np.ndarray,
    field: np.ndarray,
    lower_x: float,
    lower_y: float,
    cell_size_x: float,
    cell_size_y: float,
    num_x: int,
    num_y: int,
    periodic_x: int,
    periodic_y: int,
    degree: int,
    form: int,
) -> None:
    """The scatter of a k-form (`mpifield.kernels.spline_forms.scatter_forms_spline`)."""
    scatter_forms_spline(
        positions,
        values,
        field,
        np.array([lower_x, lower_y]),
        np.array([cell_size_x, cell_size_y]),
        np.array([num_x, num_y]),
        np.array([periodic_x, periodic_y]).astype(bool),
        degree,
        form,
    )
