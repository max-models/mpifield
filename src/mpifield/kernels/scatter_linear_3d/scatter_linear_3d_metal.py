"""The Metal version (Apple-silicon GPU, float32) of the ``scatter_linear_3d`` kernel.

``scatter_linear_3d_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker (``positions`` ``(3, N)``, ``values`` ``(components, N)``); see :mod:`mpifield.kernels._metal_grid`.
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
    int ny = grid[1], nz = grid[2];
    int ncomp = (int)values_shape[0];
    int i[2], j[2], k[2];
    float fx, fy, fz;
    mpifield_linear_cell(rel[marker], grid[0], grid[3], i[0], i[1], fx);
    mpifield_linear_cell(rel[n + marker], grid[1], grid[4], j[0], j[1], fy);
    mpifield_linear_cell(rel[2 * n + marker], grid[2], grid[5], k[0], k[1], fz);
    float wx[2] = {1.0f - fx, fx};
    float wy[2] = {1.0f - fy, fy};
    float wz[2] = {1.0f - fz, fz};
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            for (int d = 0; d < 2; ++d) {
                int row = (i[a] * ny + j[b]) * nz + k[d];
                float weight = wx[a] * wy[b] * wz[d];
                for (int c = 0; c < ncomp; ++c)
                    atomic_fetch_add_explicit(
                        &delta[row * ncomp + c], weight * values[c * n + marker],
                        memory_order_relaxed);
            }
"""

_KERNEL = make_scatter_kernel(
    "mpifield_scatter_linear_3d",
    _SOURCE,
    inputs=["rel", "values", "grid"],
    header=CELL_HEADER,
)


def scatter_linear_3d(
    positions: Array,
    values: Array,
    field: Array,
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
) -> None:
    """Trilinear deposit of ``values`` (components, N) onto ``field`` (nodes, components) on the GPU."""
    if positions.shape[1] == 0:
        return
    rel = relative(
        positions,
        (lower_x, lower_y, lower_z),
        (cell_size_x, cell_size_y, cell_size_z),
    )
    grid = [
        int(num_x),
        int(num_y),
        int(num_z),
        int(periodic_x),
        int(periodic_y),
        int(periodic_z),
    ]
    scatter(_KERNEL, rel, values, grid, field=field, n_threads=rel.shape[1])
