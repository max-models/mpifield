"""The NumPy version of the ``scatter_forms_spline_1d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``scatter_forms_spline_1d_pyccel.py``, vectorized over
the markers (`mpifield.kernels.spline_forms`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_forms import scatter_forms_spline


def scatter_forms_spline_1d(
    positions: np.ndarray,
    values: np.ndarray,
    field: np.ndarray,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
    degree: int,
    form: int,
) -> None:
    """The scatter of a k-form (`mpifield.kernels.spline_forms.scatter_forms_spline`)."""
    scatter_forms_spline(
        positions,
        values,
        field,
        np.array([lower_x]),
        np.array([cell_size_x]),
        np.array([num_x]),
        np.array([periodic_x]).astype(bool),
        degree,
        form,
    )
