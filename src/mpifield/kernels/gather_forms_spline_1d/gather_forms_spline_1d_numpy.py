"""The NumPy version of the ``gather_forms_spline_1d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_forms_spline_1d_pyccel.py``, vectorized over
the markers (`mpifield.kernels.spline_forms`).
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_forms import gather_forms_spline


def gather_forms_spline_1d(
    positions: np.ndarray,
    field: np.ndarray,
    result: np.ndarray,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
    degree: int,
    form: int,
) -> None:
    """The gather of a k-form (`mpifield.kernels.spline_forms.gather_forms_spline`)."""
    gather_forms_spline(
        positions,
        field,
        result,
        np.array([lower_x]),
        np.array([cell_size_x]),
        np.array([num_x]),
        np.array([periodic_x]),
        degree,
        form,
    )
