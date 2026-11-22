"""The field of an edge field at the markers, with the spline edge functions of degree 1 to 3, in 2D.

The Pyccel version (CPU) of the ``gather_edges_spline_2d`` kernel; ``gather_edges_spline_2d_cuda.cu``
next to this file is the CUDA version (GPU), with the same arguments. The edge of node
``k`` along axis ``a`` has the function ``B_(p-1)(r_a - k_a - 1/2) prod_(b != a) B_p(r_b - k_b)``
(``spline_edges.py``); the field is ``E_a = sum W e / h_a``. Positions are clamped on bounded
axes. With OpenMP the markers are split over the threads in contiguous chunks.

``positions`` and ``result`` are ``(2, N)``, ``edges`` ``(*num, 2)``.
"""

import numpy as np
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

from mpifield.kernels.spline_cell_pyccel import spline_cell
from mpifield.kernels.spline_edges_pyccel import along_stencil

# Fewest markers worth starting the threads for.
MIN_PARALLEL_MARKERS = 4096


def _gather_chunk_2d(
    positions: "float[:,:]",
    edges: "float[:,:,:]",
    result: "float[:,:]",
    first: int,
    last: int,
    lower_x: float,
    lower_y: float,
    cell_size_x: float,
    cell_size_y: float,
    num_x: int,
    num_y: int,
    periodic_x: int,
    periodic_y: int,
    degree: int,
) -> None:
    """The field at the markers ``first`` to ``last``."""
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    edge_x = np.zeros(4, dtype=int)
    along_x = np.zeros(4)
    exists_x = np.zeros(4, dtype=int)
    node_x = np.zeros(4, dtype=int)
    across_x = np.zeros(4)
    edge_y = np.zeros(4, dtype=int)
    along_y = np.zeros(4)
    exists_y = np.zeros(4, dtype=int)
    node_y = np.zeros(4, dtype=int)
    across_y = np.zeros(4)
    for marker in range(first, last):
        rel_x = (positions[0, marker] - lower_x) * inverse_x
        if periodic_x != 1:
            if rel_x < 0.0:
                rel_x = 0.0
            if rel_x > num_x - 1:
                rel_x = num_x - 1.0
        rel_y = (positions[1, marker] - lower_y) * inverse_y
        if periodic_y != 1:
            if rel_y < 0.0:
                rel_y = 0.0
            if rel_y > num_y - 1:
                rel_y = num_y - 1.0
        along_stencil(rel_x, num_x, periodic_x, degree, edge_x, along_x, exists_x)
        spline_cell(rel_y, num_y, periodic_y, degree, node_y, across_y)
        total = 0.0
        for a in range(degree):
            if exists_x[a] == 1:
                for b in range(degree + 1):
                    total += along_x[a] * across_y[b] * edges[edge_x[a], node_y[b], 0]
        result[0, marker] = total * inverse_x
        spline_cell(rel_x, num_x, periodic_x, degree, node_x, across_x)
        along_stencil(rel_y, num_y, periodic_y, degree, edge_y, along_y, exists_y)
        total = 0.0
        for a in range(degree + 1):
            for b in range(degree):
                if exists_y[b] == 1:
                    total += across_x[a] * along_y[b] * edges[node_x[a], edge_y[b], 1]
        result[1, marker] = total * inverse_y


def gather_edges_spline_2d(
    positions: "float[:,:]",
    edges: "float[:,:,:]",
    result: "float[:,:]",
    lower_x: float,
    lower_y: float,
    cell_size_x: float,
    cell_size_y: float,
    num_x: int,
    num_y: int,
    periodic_x: int,
    periodic_y: int,
    degree: int,
) -> None:
    """Write the field of ``edges`` at ``positions`` into ``result``."""
    num_markers = positions.shape[1]
    num_threads = int(omp_get_max_threads())
    if num_threads == 1 or num_markers < MIN_PARALLEL_MARKERS:
        _gather_chunk_2d(
            positions,
            edges,
            result,
            0,
            num_markers,
            lower_x,
            lower_y,
            cell_size_x,
            cell_size_y,
            num_x,
            num_y,
            periodic_x,
            periodic_y,
            degree,
        )
    else:
        chunk = (num_markers + num_threads - 1) // num_threads
        # fmt: off
        #$ omp parallel num_threads(num_threads) private(thread, first, last)
        # fmt: on
        thread = int(omp_get_thread_num())
        first = min(thread * chunk, num_markers)
        last = min(first + chunk, num_markers)
        _gather_chunk_2d(
            positions,
            edges,
            result,
            first,
            last,
            lower_x,
            lower_y,
            cell_size_x,
            cell_size_y,
            num_x,
            num_y,
            periodic_x,
            periodic_y,
            degree,
        )
        # fmt: off
        #$ omp end parallel
        # fmt: on
