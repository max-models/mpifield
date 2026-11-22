"""The deposit of a k-form of B-spline degree 1 to 3 from marker values, in 2D.

The Pyccel version (CPU) of the ``scatter_forms_spline_2d`` kernel; ``scatter_forms_spline_2d_cuda.cu``
next to this file is the CUDA version (GPU), with the same arguments. The transpose of
``gather_forms_spline_2d``. With OpenMP the markers are split over the threads in
contiguous chunks, each depositing into its own copy of the grid, summed in thread order.
A marker whose values are all zero deposits nothing.

``positions`` is ``(2, N)``, ``values`` ``(components, N)`` and ``field`` ``(nx, ny, components)``.
"""

import numpy as np
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

from mpifield.kernels.spline_cell_pyccel import spline_cell
from mpifield.kernels.spline_edges_pyccel import along_stencil, staggered

# Fewest markers worth the per-thread grid copies.
MIN_PARALLEL_MARKERS = 20000
# Most values of all the per-thread grid copies together (128 MB).
MAX_BUFFER_VALUES = 16777216


def _scatter_chunk_2d(
    positions: "float[:,:]",
    values: "float[:,:]",
    buffer: "float[:,:,:]",
    offset: int,
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
    form: int,
) -> None:
    """Deposit the markers ``first`` to ``last`` into the x-rows ``offset`` onward of ``buffer``."""
    num_components = values.shape[0]
    inverse_x = 1.0 / cell_size_x
    index_x = np.zeros(4, dtype=int)
    weight_x = np.zeros(4)
    exists_x = np.zeros(4, dtype=int)
    inverse_y = 1.0 / cell_size_y
    index_y = np.zeros(4, dtype=int)
    weight_y = np.zeros(4)
    exists_y = np.zeros(4, dtype=int)
    for marker in range(first, last):
        deposit = False
        for component in range(num_components):
            if values[component, marker] != 0.0:
                deposit = True
        if not deposit:
            continue
        rel_x = (positions[0, marker] - lower_x) * inverse_x
        if periodic_x == 0:
            if rel_x < 0.0:
                rel_x = 0.0
            if rel_x > num_x - 1:
                rel_x = num_x - 1.0
        rel_y = (positions[1, marker] - lower_y) * inverse_y
        if periodic_y == 0:
            if rel_y < 0.0:
                rel_y = 0.0
            if rel_y > num_y - 1:
                rel_y = num_y - 1.0
        for component in range(num_components):
            value = values[component, marker]
            if staggered(form, 2, component, 0) == 1:
                along_stencil(
                    rel_x, num_x, periodic_x, degree, index_x, weight_x, exists_x
                )
                size_x = degree
            else:
                spline_cell(rel_x, num_x, periodic_x, degree, index_x, weight_x)
                for i in range(degree + 1):
                    exists_x[i] = 1
                size_x = degree + 1
            if staggered(form, 2, component, 1) == 1:
                along_stencil(
                    rel_y, num_y, periodic_y, degree, index_y, weight_y, exists_y
                )
                size_y = degree
            else:
                spline_cell(rel_y, num_y, periodic_y, degree, index_y, weight_y)
                for i in range(degree + 1):
                    exists_y[i] = 1
                size_y = degree + 1
            for a in range(size_x):
                if exists_x[a] == 1:
                    for b in range(size_y):
                        if exists_y[b] == 1:
                            buffer[offset + index_x[a], index_y[b], component] += (
                                weight_x[a] * weight_y[b] * value
                            )


def scatter_forms_spline_2d(
    positions: "float[:,:]",
    values: "float[:,:]",
    field: "float[:,:,:]",
    lower_x: float,
    lower_y: float,
    cell_size_x: float,
    cell_size_y: float,
    num_x: int,
    num_y: int,
    periodic_x: int,
    periodic_y: int,
    degree: int,
    form: int,
) -> None:
    """Add ``W values`` at ``positions`` to ``field``."""
    num_markers = positions.shape[1]
    num_components = values.shape[0]
    num_threads = int(omp_get_max_threads())
    grid_values = num_x * num_y * num_components
    if (
        num_threads == 1
        or num_markers < MIN_PARALLEL_MARKERS
        or num_markers < grid_values
        or num_threads * grid_values > MAX_BUFFER_VALUES
    ):
        _scatter_chunk_2d(
            positions,
            values,
            field,
            0,
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
            form,
        )
    else:
        buffer = np.zeros((num_threads * num_x, num_y, num_components))
        chunk = (num_markers + num_threads - 1) // num_threads
        # fmt: off
        #$ omp parallel num_threads(num_threads) private(thread, first, last)
        # fmt: on
        thread = int(omp_get_thread_num())
        first = min(thread * chunk, num_markers)
        last = min(first + chunk, num_markers)
        _scatter_chunk_2d(
            positions,
            values,
            buffer,
            thread * num_x,
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
            form,
        )
        # fmt: off
        #$ omp end parallel
        #$ omp parallel for private(j, component, total, thread)
        # fmt: on
        for i in range(num_x):
            for j in range(num_y):
                for component in range(num_components):
                    total = 0.0
                    for thread in range(num_threads):
                        total += buffer[thread * num_x + i, j, component]
                    field[i, j, component] += total
