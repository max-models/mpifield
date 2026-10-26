"""The NumPy version of the ``scatter_linear_1d`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``scatter_linear_1d_pyccel.py``, vectorized
over the markers.
"""

from __future__ import annotations

import numpy as np


def scatter_linear_1d(
    positions: np.ndarray,
    values: np.ndarray,
    field: np.ndarray,
    lower_x: float,
    cell_size: float,
    num_gridpoints: int,
) -> None:
    """Deposit scalar marker values onto ``field`` with linear weights (periodic layout)."""
    relative_x = (positions - lower_x) / cell_size
    left = np.clip(np.floor(relative_x).astype(np.intp), 0, num_gridpoints - 1)
    fraction = np.clip(relative_x - left, 0.0, 1.0)
    right = (left + 1) % num_gridpoints
    # bincount sums repeated cells in compiled code, much faster than np.add.at
    field += np.bincount(
        left, weights=(1.0 - fraction) * values, minlength=field.shape[0]
    )
    field += np.bincount(right, weights=fraction * values, minlength=field.shape[0])
