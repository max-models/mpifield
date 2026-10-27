"""The Metal version (Apple-silicon GPU, float32) of the ``scatter_quadratic_3d`` kernel.

``scatter_quadratic_3d_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker (``positions`` ``(3, N)``, ``values``
``(components, N)``); see :mod:`mpifield.kernels._metal_grid`.
"""

from __future__ import annotations

import numpy as np

from mpifield._types import Array
from mpifield.kernels._metal_grid import (
    QUADRATIC_HEADER,
    make_scatter_kernel,
    relative,
    scatter,
)

_SOURCE = """
    uint marker = thread_position_in_grid.x;
    uint n = rel_shape[1];
    if (marker >= n) return;
    int ncomp = (int)values_shape[0];
    // dead markers have zero weight: skip them without reading their positions
    bool deposit = false;
    for (int c = 0; c < ncomp; ++c) {
        if (values[c * n + marker] != 0.0f) deposit = true;
    }
    if (!deposit) return;
    int ny = grid[1], nz = grid[2];
    int node_x[3];
    float weight_x[3];
    int node_y[3];
    float weight_y[3];
    int node_z[3];
    float weight_z[3];
    mpifield_quadratic_cell(rel[marker], grid[0], grid[3], node_x, weight_x);
    mpifield_quadratic_cell(rel[n + marker], grid[1], grid[4], node_y, weight_y);
    mpifield_quadratic_cell(rel[2 * n + marker], grid[2], grid[5], node_z, weight_z);
    for (int c = 0; c < ncomp; ++c) {
        float value = values[c * n + marker];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int e = 0; e < 3; ++e)
                    atomic_fetch_add_explicit(&delta[((node_x[a] * ny + node_y[b]) * nz + node_z[e]) * ncomp + c],
                                              weight_x[a] * weight_y[b] * weight_z[e] * value, memory_order_relaxed);
    }
"""

_KERNEL = make_scatter_kernel(
    "mpifield_scatter_quadratic_3d",
    _SOURCE,
    inputs=["rel", "values", "grid"],
    header=QUADRATIC_HEADER,
)


def scatter_quadratic_3d(
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
    """Quadratic B-spline deposit of ``values`` (components, N) onto ``field`` on the GPU."""
    if positions.shape[1] == 0:
        return
    rel = relative(
        positions, (lower_x, lower_y, lower_z), (cell_size_x, cell_size_y, cell_size_z)
    )
    # dead markers may sit at non-finite positions: keep them out of float32
    rel = np.where(np.any(np.asarray(values) != 0.0, axis=0), rel, 0.0)
    grid = [
        int(num_x),
        int(num_y),
        int(num_z),
        int(periodic_x),
        int(periodic_y),
        int(periodic_z),
    ]
    scatter(_KERNEL, rel, values, grid, field=field, n_threads=rel.shape[1])
