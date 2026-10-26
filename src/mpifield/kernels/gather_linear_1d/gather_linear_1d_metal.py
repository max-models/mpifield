"""The Metal version (Apple-silicon GPU, float32) of the ``gather_linear_1d`` kernel.

``gather_linear_1d_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker (``positions`` ``(1, N)``, ``result`` ``(components, N)``); see :mod:`mpifield.kernels._metal_grid`.
"""

from __future__ import annotations

import numpy as np

from mpifield._types import Array
from mpifield.kernels._metal_grid import CELL_HEADER, make_kernel, relative

_SOURCE = """
    uint marker = thread_position_in_grid.x;
    uint n = rel_shape[0];
    if (marker >= n) return;
    int ncomp = (int)field_shape[1];
    int i0, i1;
    float fx;
    mpifield_linear_cell(rel[marker], grid[0], grid[1], i0, i1, fx);
    for (int c = 0; c < ncomp; ++c) {
        result[c * n + marker] =
            (1.0f - fx) * field[i0 * ncomp + c] + fx * field[i1 * ncomp + c];
    }
"""

_KERNEL = make_kernel(
    "mpifield_gather_linear_1d",
    _SOURCE,
    inputs=["rel", "field", "grid"],
    outputs=["result"],
    header=CELL_HEADER,
)


def gather_linear_1d(
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
    rel = np.ascontiguousarray(relative(positions, (lower_x,), (cell_size_x,))[0])
    _KERNEL(
        rel, field, [int(num_x), int(periodic_x)], out=result, n_threads=rel.shape[0]
    )
