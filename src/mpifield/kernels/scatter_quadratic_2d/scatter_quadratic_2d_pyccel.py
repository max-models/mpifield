"""Quadratic B-spline deposit of marker values onto a 2D grid, any number of components.

The Pyccel version (CPU) of the ``scatter_quadratic_2d`` kernel; ``scatter_quadratic_2d_cuda.cu`` next to
this file is the CUDA version (GPU), with the same arguments.  Both are loaded
by :mod:`mpifield.kernels.catalog`; this file is compiled with Pyccel there.

Each marker deposits onto its nearest node and the two around it along each axis
(``quadratic_cell.cuh``). With OpenMP the markers are split over the threads in
contiguous chunks, each depositing into its own copy of the grid, and the copies
are summed into ``field`` in thread order: no atomic updates, and the result is
reproducible for a given number of threads (the summation order, hence the last
bits, differs from the serial one). Few markers, or copies that would together
hold more than ``MAX_BUFFER_VALUES`` values, are deposited serially.

``positions`` is ``(2, N)``, ``values`` ``(components, N)`` and ``field``
``(nx, ny, components)``.
"""

import numpy as np
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

# Fewest markers worth the per-thread grid copies.
MIN_PARALLEL_MARKERS = 20000
# Most values of all the per-thread grid copies together (128 MB).
MAX_BUFFER_VALUES = 16777216


def _quadratic_cell(
    rel: float, num: int, periodic: int, node: "int[:]", weight: "float[:]"
) -> None:
    """The stencil nodes and weights of ``rel`` along one axis (``quadratic_cell.cuh``)."""
    # comparisons instead of min/max, which Pyccel turns into variadic C
    # functions that the compiler cannot inline
    r = rel
    if periodic == 1:
        if not (r >= 0.0 and r < num):
            r = r - num * (r // num)
        centre = int(r + 0.5)
        node[0] = (centre + num - 1) % num
        node[1] = centre % num
        node[2] = (centre + 1) % num
    else:
        if r < 0.0:
            r = 0.0
        if r > num - 1:
            r = num - 1.0
        centre = int(r + 0.5)
        node[0] = centre - 1 if centre > 0 else 0
        node[1] = centre
        node[2] = centre + 1 if centre < num - 1 else num - 1
    offset = r - centre
    weight[0] = 0.5 * (0.5 - offset) * (0.5 - offset)
    weight[1] = 0.75 - offset * offset
    weight[2] = 0.5 * (0.5 + offset) * (0.5 + offset)


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
) -> None:
    """Deposit the markers ``first`` to ``last`` into the x-rows ``offset`` onward of ``buffer``."""
    num_components = values.shape[0]
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    node_x = np.zeros(3, dtype=int)
    weight_x = np.zeros(3)
    node_y = np.zeros(3, dtype=int)
    weight_y = np.zeros(3)
    for marker in range(first, last):
        # a zero-weight marker (a dead one left in storage) deposits nothing,
        # whatever its position
        deposit = False
        for component in range(num_components):
            if values[component, marker] != 0.0:
                deposit = True
        if not deposit:
            continue
        _quadratic_cell(
            (positions[0, marker] - lower_x) * inverse_x,
            num_x,
            periodic_x,
            node_x,
            weight_x,
        )
        _quadratic_cell(
            (positions[1, marker] - lower_y) * inverse_y,
            num_y,
            periodic_y,
            node_y,
            weight_y,
        )
        for a in range(3):
            row = offset + node_x[a]
            for b in range(3):
                weight = weight_x[a] * weight_y[b]
                for component in range(num_components):
                    buffer[row, node_y[b], component] += (
                        weight * values[component, marker]
                    )


def scatter_quadratic_2d(
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
) -> None:
    """Deposit ``values`` (components, N) onto ``field`` (nx, ny, components)."""
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
