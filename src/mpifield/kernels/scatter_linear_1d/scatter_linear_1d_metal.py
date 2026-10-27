"""The Metal version (Apple-silicon GPU, float32) of the ``scatter_linear_1d`` kernel.

``scatter_linear_1d_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker; see :mod:`mpifield.kernels._metal_grid`.
"""

from __future__ import annotations

import numpy as np

from mpifield._types import Array
from mpifield.kernels._metal_grid import CELL_HEADER, make_scatter_kernel, scatter

_SOURCE = """
    uint marker = thread_position_in_grid.x;
    if (marker >= rel_shape[0]) return;
    float value = values[marker];
    // dead markers have zero weight: skip them without reading their positions
    if (value == 0.0f) return;
    int left, right;
    float fraction;
    mpifield_linear_cell(rel[marker], grid[0], grid[1], left, right, fraction);
    atomic_fetch_add_explicit(&delta[left], (1.0f - fraction) * value, memory_order_relaxed);
    atomic_fetch_add_explicit(&delta[right], fraction * value, memory_order_relaxed);
"""

_KERNEL = make_scatter_kernel(
    "mpifield_scatter_linear_1d",
    _SOURCE,
    inputs=["rel", "values", "grid"],
    header=CELL_HEADER,
)


def scatter_linear_1d(
    positions: Array,
    values: Array,
    field: Array,
    lower_x: float,
    cell_size: float,
    num_gridpoints: int,
    periodic_x: int,
) -> None:
    """Deposit scalar marker values onto a 1D grid with linear weights on the GPU."""
    if positions.shape[0] == 0:
        return
    rel = (np.asarray(positions, dtype=np.float64) - lower_x) / cell_size
    # dead markers may sit at non-finite positions: keep them out of float32
    rel = np.where(np.asarray(values) == 0.0, 0.0, rel)
    grid = [int(num_gridpoints), int(periodic_x)]
    scatter(_KERNEL, rel, values, grid, field=field, n_threads=rel.shape[0])
