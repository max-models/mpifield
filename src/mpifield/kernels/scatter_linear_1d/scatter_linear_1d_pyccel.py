"""Linear (cloud-in-cell) deposit of marker values onto a 1D grid.

The Pyccel version (CPU) of the ``scatter_linear_1d`` kernel; ``scatter_linear_1d_cuda.cu`` next to
this file is the CUDA version (GPU), with the same arguments.  Both are loaded
by :mod:`mpifield.kernels.catalog`; this file is compiled with Pyccel there.

With OpenMP the markers are split over the threads, each depositing into its own
copy of the grid, and the copies are summed into ``field``: no atomic updates, and
the result is reproducible for a given number of threads (the summation order, hence
the last bits, differs from the serial one). Few markers are deposited serially.
"""

import numpy as np
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

# Fewest markers worth the per-thread grid copies.
MIN_PARALLEL_MARKERS = 20000


def _scatter_chunk_1d(
    positions: "float[:]",
    values: "float[:]",
    buffer: "float[:]",
    offset: int,
    first: int,
    last: int,
    lower_x: float,
    cell_size: float,
    num_gridpoints: int,
    periodic_x: int,
) -> None:
    """Deposit the markers ``first`` to ``last`` into the nodes ``offset`` onward of ``buffer``."""
    # comparisons instead of min/max, which Pyccel turns into variadic C
    # functions that the compiler cannot inline
    for marker in range(first, last):
        # a zero-weight marker (a dead one left in storage) deposits nothing,
        # whatever its position
        if values[marker] == 0.0:
            continue
        relative_x = (positions[marker] - lower_x) / cell_size
        if periodic_x == 1:
            if not (relative_x >= 0.0 and relative_x < num_gridpoints):
                relative_x = relative_x - num_gridpoints * (
                    relative_x // num_gridpoints
                )
            left = int(relative_x)
            if left >= num_gridpoints:
                left = num_gridpoints - 1
            right = left + 1
            if right == num_gridpoints:
                right = 0
        else:
            left = int(relative_x) if relative_x >= 0.0 else 0
            if left > num_gridpoints - 2:
                left = num_gridpoints - 2
            right = left + 1
        fraction = relative_x - left
        if fraction < 0.0:
            fraction = 0.0
        if fraction > 1.0:
            fraction = 1.0
        buffer[offset + left] += (1.0 - fraction) * values[marker]
        buffer[offset + right] += fraction * values[marker]


def scatter_linear_1d(
    positions: "float[:]",
    values: "float[:]",
    field: "float[:]",
    lower_x: float,
    cell_size: float,
    num_gridpoints: int,
    periodic_x: int,
) -> None:
    """Deposit scalar marker values onto a 1D grid with linear weights.

    On a periodic axis (``periodic_x`` 1) the right neighbour of the last node is
    the first node and positions wrap into the period; on a bounded one a marker
    outside the grid uses the edge cell, its weight clamped to [0, 1].
    """
    num_markers = positions.shape[0]
    num_threads = int(omp_get_max_threads())
    if (
        num_threads == 1
        or num_markers < MIN_PARALLEL_MARKERS
        or num_markers < field.shape[0]
    ):
        _scatter_chunk_1d(
            positions,
            values,
            field,
            0,
            0,
            num_markers,
            lower_x,
            cell_size,
            num_gridpoints,
            periodic_x,
        )
    else:
        buffer = np.zeros(num_threads * num_gridpoints)
        chunk = (num_markers + num_threads - 1) // num_threads
        # fmt: off
        #$ omp parallel num_threads(num_threads) private(thread, first, last)
        # fmt: on
        thread = int(omp_get_thread_num())
        first = thread * chunk
        last = min(first + chunk, num_markers)
        _scatter_chunk_1d(
            positions,
            values,
            buffer,
            thread * num_gridpoints,
            first,
            last,
            lower_x,
            cell_size,
            num_gridpoints,
            periodic_x,
        )
        # fmt: off
        #$ omp end parallel
        #$ omp parallel for private(thread, total)
        # fmt: on
        for node in range(field.shape[0]):
            total = 0.0
            for thread in range(num_threads):
                total += buffer[thread * num_gridpoints + node]
            field[node] += total
