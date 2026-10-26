"""The Metal version (Apple-silicon GPU, float32) of the ``scatter_linear_2d`` kernel.

``scatter_linear_2d_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker (``positions`` ``(2, N)``, ``values`` ``(components, N)``); see :mod:`mpifield.kernels._metal_grid`.
"""

from __future__ import annotations

from mpifield._types import Array
from mpifield.kernels._metal_grid import (
    CELL_HEADER,
    make_scatter_kernel,
    relative,
    scatter,
)

_SOURCE = """
    uint marker = thread_position_in_grid.x;
    uint n = rel_shape[1];
    if (marker >= n) return;
    int ny = grid[1];
    int ncomp = (int)values_shape[0];
    int i0, i1, j0, j1;
    float fx, fy;
    mpifield_linear_cell(rel[marker], grid[0], grid[2], i0, i1, fx);
    mpifield_linear_cell(rel[n + marker], grid[1], grid[3], j0, j1, fy);
    float w_ll = (1.0f - fx) * (1.0f - fy);
    float w_lr = fx * (1.0f - fy);
    float w_ul = (1.0f - fx) * fy;
    float w_ur = fx * fy;
    for (int c = 0; c < ncomp; ++c) {
        float value = values[c * n + marker];
        atomic_fetch_add_explicit(&delta[(i0 * ny + j0) * ncomp + c], w_ll * value, memory_order_relaxed);
        atomic_fetch_add_explicit(&delta[(i1 * ny + j0) * ncomp + c], w_lr * value, memory_order_relaxed);
        atomic_fetch_add_explicit(&delta[(i0 * ny + j1) * ncomp + c], w_ul * value, memory_order_relaxed);
        atomic_fetch_add_explicit(&delta[(i1 * ny + j1) * ncomp + c], w_ur * value, memory_order_relaxed);
    }
"""

_KERNEL = make_scatter_kernel(
    "mpifield_scatter_linear_2d",
    _SOURCE,
    inputs=["rel", "values", "grid"],
    header=CELL_HEADER,
)


def scatter_linear_2d(
    positions: Array,
    values: Array,
    field: Array,
    lower_x: float,
    lower_y: float,
    cell_size_x: float,
    cell_size_y: float,
    num_x: int,
    num_y: int,
    periodic_x: int,
    periodic_y: int,
) -> None:
    """Bilinear deposit of ``values`` (components, N) onto ``field`` on the GPU."""
    if positions.shape[1] == 0:
        return
    rel = relative(
        positions,
        (lower_x, lower_y),
        (cell_size_x, cell_size_y),
    )
    grid = [int(num_x), int(num_y), int(periodic_x), int(periodic_y)]
    scatter(_KERNEL, rel, values, grid, field=field, n_threads=rel.shape[1])
