"""The Metal version (Apple-silicon GPU, float32) of the ``gather_quadratic_2d`` kernel.

``gather_quadratic_2d_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker (``positions`` ``(2, N)``, ``result``
``(components, N)``); see :mod:`mpifield.kernels._metal_grid`.
"""

from __future__ import annotations

from mpifield._types import Array
from mpifield.kernels._metal_grid import QUADRATIC_HEADER, make_kernel, relative

_SOURCE = """
    uint marker = thread_position_in_grid.x;
    uint n = rel_shape[1];
    if (marker >= n) return;
    int ncomp = (int)field_shape[2];
    int ny = grid[1];
    int node_x[3];
    float weight_x[3];
    int node_y[3];
    float weight_y[3];
    mpifield_quadratic_cell(rel[marker], grid[0], grid[2], node_x, weight_x);
    mpifield_quadratic_cell(rel[n + marker], grid[1], grid[3], node_y, weight_y);
    for (int c = 0; c < ncomp; ++c) {
        float total = 0.0f;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                total += weight_x[a] * weight_y[b] * field[(node_x[a] * ny + node_y[b]) * ncomp + c];
        result[c * n + marker] = total;
    }
"""

_KERNEL = make_kernel(
    "mpifield_gather_quadratic_2d",
    _SOURCE,
    inputs=["rel", "field", "grid"],
    outputs=["result"],
    header=QUADRATIC_HEADER,
)


def gather_quadratic_2d(
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
    """Interpolate ``field`` (nx, ny, components) at the markers into ``result`` (components, N)."""
    if positions.shape[1] == 0:
        return
    rel = relative(positions, (lower_x, lower_y), (cell_size_x, cell_size_y))
    grid = [int(num_x), int(num_y), int(periodic_x), int(periodic_y)]
    _KERNEL(rel, field, grid, out=result, n_threads=rel.shape[1])
