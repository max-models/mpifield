"""Helpers of the Metal gather, scatter and deposit kernels.

Positions ``(ndim, N)`` are turned into cell coordinates ``(x - lower) / cell_size`` on
the host in float64; the kernels compute the cell and the weights in float32 with
:data:`CELL_HEADER`, the arithmetic of ``linear_cell.cuh`` (:data:`QUADRATIC_HEADER` and
``quadratic_cell.cuh`` for the quadratic ones). A scatter accumulates
into a zeroed float32 grid with float atomics (no 64-bit atomics on the Apple GPU),
which the host adds to the caller's grid in float64: the summation order, hence the
last bits, differ from the CPU. See :mod:`mpifield.kernels`.
"""

from __future__ import annotations

import cunumpy as xp
import numpy as np

from mpifield._types import Array
from mpifield.kernels import KERNEL_PACKAGE

__all__ = [
    "CELL_HEADER",
    "QUADRATIC_HEADER",
    "make_scatter_kernel",
    "relative",
    "scatter",
]

#: The cell of a coordinate along one axis, as ``mpifield_linear_cell`` in ``linear_cell.cuh``.
CELL_HEADER = """
inline void mpifield_linear_cell(
    float rel, int num, int periodic,
    thread int& left, thread int& right, thread float& fraction) {
    int l;
    if (periodic) {
        rel = rel - (float)num * metal::floor(rel / (float)num);
        l = ((int)rel) % num;
        right = (l + 1) % num;
    } else {
        l = rel >= 0.0f ? (int)rel : -1;
        if (l < 0) l = 0;
        if (l > num - 2) l = num - 2;
        right = l + 1;
    }
    left = l;
    fraction = metal::clamp(rel - (float)l, 0.0f, 1.0f);
}
"""

#: The stencil of a coordinate along one axis for quadratic weights, as
#: ``mpifield_quadratic_cell`` in ``quadratic_cell.cuh``.
QUADRATIC_HEADER = """
inline void mpifield_quadratic_cell(
    float rel, int num, int periodic, thread int* node, thread float* weight) {
    int c;
    if (periodic) {
        rel = rel - (float)num * metal::floor(rel / (float)num);
        c = (int)(rel + 0.5f);
        node[0] = (c + num - 1) % num;
        node[1] = c % num;
        node[2] = (c + 1) % num;
    } else {
        rel = metal::clamp(rel, 0.0f, (float)(num - 1));
        c = (int)(rel + 0.5f);
        node[0] = c > 0 ? c - 1 : 0;
        node[1] = c;
        node[2] = c < num - 1 ? c + 1 : num - 1;
    }
    float d = rel - (float)c;
    weight[0] = 0.5f * (0.5f - d) * (0.5f - d);
    weight[1] = 0.75f - d * d;
    weight[2] = 0.5f * (0.5f + d) * (0.5f + d);
}
"""


def relative(
    positions: Array, lower: tuple[float, ...], cell: tuple[float, ...]
) -> Array:
    """Return the cell coordinates ``(x - lower) / cell_size``, shape ``(ndim, N)``.

    ``positions`` is component-major, ``(ndim, N)``; the result has the same layout,
    so a Metal kernel reads component ``d`` of marker ``i`` as ``rel[d * n + i]``.
    """
    positions = np.asarray(positions)
    offsets = np.asarray(lower, dtype=np.float64)
    sizes = np.asarray(cell, dtype=np.float64)
    return (positions.reshape(-1, positions.shape[-1]) - offsets[:, None]) / sizes[
        :, None
    ]


def make_scatter_kernel(
    name: str, source: str, inputs: list[str], header: str = ""
) -> xp.kernels.MetalKernel:
    """A kernel adding into the zeroed float32 output ``delta`` with float atomics."""
    return KERNEL_PACKAGE.metal_kernel(
        name, source, inputs, ["delta"], header=header, atomic_outputs=True
    )


def scatter(
    kernel: xp.kernels.MetalKernel,
    *args: object,
    field: Array,
    n_threads: int,
) -> None:
    """Run a scatter ``kernel`` and add its float32 result to ``field`` (in float64)."""
    delta = np.zeros(field.shape, dtype=np.float64)
    kernel(*args, out=delta, n_threads=max(n_threads, 1))
    field += delta
