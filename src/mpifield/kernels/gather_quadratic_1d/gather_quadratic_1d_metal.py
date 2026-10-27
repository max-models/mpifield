"""The Metal version (Apple-silicon GPU, float32) of the ``gather_quadratic_1d`` kernel.

``gather_quadratic_1d_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker (``positions`` ``(1, N)``, ``result``
``(components, N)``); see :mod:`mpifield.kernels._metal_grid`.
"""

from __future__ import annotations

from mpifield._types import Array
from mpifield.kernels import KERNEL_PACKAGE
from mpifield.kernels._metal_grid import QUADRATIC_HEADER, relative

_SOURCE = """
    uint marker = thread_position_in_grid.x;
    uint n = rel_shape[1];
    if (marker >= n) return;
    int ncomp = (int)field_shape[1];
    int node_x[3];
    float weight_x[3];
    mpifield_quadratic_cell(rel[marker], grid[0], grid[1], node_x, weight_x);
    for (int c = 0; c < ncomp; ++c) {
        float total = 0.0f;
        for (int a = 0; a < 3; ++a)
            total += weight_x[a] * field[node_x[a] * ncomp + c];
        result[c * n + marker] = total;
    }
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "mpifield_gather_quadratic_1d",
    _SOURCE,
    inputs=["rel", "field", "grid"],
    outputs=["result"],
    header=QUADRATIC_HEADER,
)


def gather_quadratic_1d(
    positions: Array,
    field: Array,
    result: Array,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> None:
    """Interpolate ``field`` (nx, components) at the markers into ``result`` (components, N)."""
    if positions.shape[1] == 0:
        return
    rel = relative(positions, (lower_x,), (cell_size_x,))
    grid = [int(num_x), int(periodic_x)]
    _KERNEL(rel, field, grid, out=result, n_threads=rel.shape[1])
