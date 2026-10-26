"""The Metal version (Apple-silicon GPU, float32) of the ``gather_linear_2d`` kernel.

``gather_linear_2d_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker (``positions`` ``(2, N)``, ``result`` ``(components, N)``); see :mod:`mpifield.kernels._metal_grid`.
"""

from __future__ import annotations

from mpifield._types import Array
from mpifield.kernels._metal_grid import CELL_HEADER, make_kernel, relative

_SOURCE = """
    uint marker = thread_position_in_grid.x;
    uint n = rel_shape[1];
    if (marker >= n) return;
    int ny = grid[1];
    int ncomp = (int)field_shape[2];
    int i0, i1, j0, j1;
    float fx, fy;
    mpifield_linear_cell(rel[marker], grid[0], grid[2], i0, i1, fx);
    mpifield_linear_cell(rel[n + marker], grid[1], grid[3], j0, j1, fy);
    float w_ll = (1.0f - fx) * (1.0f - fy);
    float w_lr = fx * (1.0f - fy);
    float w_ul = (1.0f - fx) * fy;
    float w_ur = fx * fy;
    for (int c = 0; c < ncomp; ++c) {
        result[c * n + marker] = w_ll * field[(i0 * ny + j0) * ncomp + c]
                                   + w_lr * field[(i1 * ny + j0) * ncomp + c]
                                   + w_ul * field[(i0 * ny + j1) * ncomp + c]
                                   + w_ur * field[(i1 * ny + j1) * ncomp + c];
    }
"""

_KERNEL = make_kernel(
    "mpifield_gather_linear_2d",
    _SOURCE,
    inputs=["rel", "field", "grid"],
    outputs=["result"],
    header=CELL_HEADER,
)


def gather_linear_2d(
    positions: Array,
    field: Array,
    result: Array,
    lower_x: float,
    lower_y: float,
    cell_size_x: float,
    cell_size_y: float,
    num_x: int,
    num_y: int,
    periodic_x: int,
    periodic_y: int,
) -> None:
    """Bilinear interpolation of ``field`` (nx, ny, components) at the markers on the GPU."""
    if positions.shape[1] == 0:
        return
    rel = relative(
        positions,
        (lower_x, lower_y),
        (cell_size_x, cell_size_y),
    )
    grid = [int(num_x), int(num_y), int(periodic_x), int(periodic_y)]
    _KERNEL(rel, field, grid, out=result, n_threads=rel.shape[1])
