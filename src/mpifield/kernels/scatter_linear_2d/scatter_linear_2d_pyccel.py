"""Bilinear deposit of marker values onto a 2D grid, any number of components.

The Pyccel version (CPU) of the ``scatter_linear_2d`` kernel; ``scatter_linear_2d_cuda.cu`` next to
this file is the CUDA version (GPU), with the same arguments.  Both are loaded
by :mod:`mpifield.kernels.catalog`; this file is compiled with Pyccel there.

With OpenMP the markers are split over the threads in contiguous chunks, each
depositing into its own buffer, and the buffers are summed into ``field``: no
atomic updates, and the result is reproducible for a given number of threads (the
summation order, hence the last bits, differs from the serial one). A thread's
buffer covers only the x-rows its markers touch: markers sorted by cell
(e.g. picnic's ``Markers.sort_by_cell``) give each thread a narrow
band of rows, so zeroing and summing the buffers costs about one grid instead of
one grid per thread; unsorted markers give each thread the whole grid, as before.
On a periodic x axis the band runs past the last row (``num_x`` is row 0 again)
and is folded back in the sum. Few markers are deposited serially.

``positions`` is ``(2, N)`` and ``values`` ``(components, N)`` (component-major).
"""

import numpy as np
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

# Fewest markers worth the per-thread buffers.
MIN_PARALLEL_MARKERS = 20000


def _scatter_chunk_2d(
    positions: "float[:,:]",
    values: "float[:,:]",
    buffer: "float[:,:,:]",
    offset: int,
    first_plane: int,
    wrap: int,
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
    """Deposit the markers ``first`` to ``last`` into the x-rows ``offset`` onward of ``buffer``.

    Row ``offset`` of ``buffer`` is x-row ``first_plane`` of the grid. With
    ``wrap`` 1 the right x-node of a periodic axis wraps to row 0 (the buffer is
    the whole grid); with 0 it is row ``num_x`` of the band.
    """
    num_components = values.shape[0]
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    for marker in range(first, last):
        # a zero-weight marker (a dead one left in storage) deposits nothing,
        # whatever its position
        if num_components == 1:
            single = values[0, marker]
            if single == 0.0:
                continue
        else:
            deposit = False
            for component in range(num_components):
                if values[component, marker] != 0.0:
                    deposit = True
            if not deposit:
                continue
        # the cell and weights along each axis, as in the 3D kernel: inverse
        # spacing, comparisons instead of min/max (calls in Pyccel's C), and
        # the floor modulo only for a periodic position outside [0, n) cells
        rel_x = (positions[0, marker] - lower_x) * inverse_x
        rel_y = (positions[1, marker] - lower_y) * inverse_y
        if periodic_x == 1:
            if not (rel_x >= 0.0 and rel_x < num_x):
                rel_x = rel_x - num_x * (rel_x // num_x)
            i_left = int(rel_x)
            if i_left >= num_x:
                i_left = num_x - 1
        else:
            i_left = int(rel_x) if rel_x >= 0.0 else 0
            if i_left > num_x - 2:
                i_left = num_x - 2
        i_right = i_left + 1
        if wrap == 1 and i_right == num_x:
            i_right = 0
        if periodic_y == 1:
            if not (rel_y >= 0.0 and rel_y < num_y):
                rel_y = rel_y - num_y * (rel_y // num_y)
            j_left = int(rel_y)
            if j_left >= num_y:
                j_left = num_y - 1
            j_right = j_left + 1
            if j_right == num_y:
                j_right = 0
        else:
            j_left = int(rel_y) if rel_y >= 0.0 else 0
            if j_left > num_y - 2:
                j_left = num_y - 2
            j_right = j_left + 1
        fx = rel_x - i_left
        if fx < 0.0:
            fx = 0.0
        if fx > 1.0:
            fx = 1.0
        fy = rel_y - j_left
        if fy < 0.0:
            fy = 0.0
        if fy > 1.0:
            fy = 1.0
        w_ll = (1.0 - fx) * (1.0 - fy)
        w_lr = fx * (1.0 - fy)
        w_ul = (1.0 - fx) * fy
        w_ur = fx * fy
        row_left = offset + i_left - first_plane
        row_right = offset + i_right - first_plane
        if num_components == 1:
            # the charge deposit: one value, no loop over components
            buffer[row_left, j_left, 0] += w_ll * single
            buffer[row_right, j_left, 0] += w_lr * single
            buffer[row_left, j_right, 0] += w_ul * single
            buffer[row_right, j_right, 0] += w_ur * single
        else:
            for component in range(num_components):
                value = values[component, marker]
                buffer[row_left, j_left, component] += w_ll * value
                buffer[row_right, j_left, component] += w_lr * value
                buffer[row_left, j_right, component] += w_ul * value
                buffer[row_right, j_right, component] += w_ur * value


def _x_rows(
    positions: "float[:,:]",
    first: int,
    last: int,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> "tuple[int, int]":
    """The first x-row and the number of x-rows the markers ``first`` to ``last`` touch.

    The band covers the left x-node of every marker (also the zero-value ones
    that deposit nothing, so the scan reads only ``positions[0]``) and the node
    after it, without the periodic wrap: up to row ``num_x``. ``(0, 0)`` for
    no markers.
    """
    lowest = num_x
    highest = -1
    inverse_x = 1.0 / cell_size_x
    for marker in range(first, last):
        # the left x-node of _scatter_chunk_2d, compared without min/max
        rel_x = (positions[0, marker] - lower_x) * inverse_x
        if periodic_x == 1:
            if not (rel_x >= 0.0 and rel_x < num_x):
                rel_x = rel_x - num_x * (rel_x // num_x)
            i_left = int(rel_x)
            if i_left >= num_x:
                i_left = num_x - 1
        else:
            i_left = int(rel_x) if rel_x >= 0.0 else 0
            if i_left > num_x - 2:
                i_left = num_x - 2
        if i_left < lowest:
            lowest = i_left
        if i_left > highest:
            highest = i_left
    if highest < 0:
        return 0, 0
    return lowest, highest + 2 - lowest


def scatter_linear_2d(
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
    if (
        num_threads == 1
        or num_markers < MIN_PARALLEL_MARKERS
        or num_markers < field.shape[0] * field.shape[1]
    ):
        _scatter_chunk_2d(
            positions,
            values,
            field,
            0,
            0,
            1,
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
        chunk = (num_markers + num_threads - 1) // num_threads
        first_rows = np.zeros(num_threads, dtype=int)
        num_rows = np.zeros(num_threads, dtype=int)
        # fmt: off
        #$ omp parallel num_threads(num_threads) private(thread, first, last, band_first, band_rows)
        # fmt: on
        thread = int(omp_get_thread_num())
        first = min(thread * chunk, num_markers)
        last = min(first + chunk, num_markers)
        band_first, band_rows = _x_rows(
            positions, first, last, lower_x, cell_size_x, num_x, periodic_x
        )
        first_rows[thread] = band_first
        num_rows[thread] = band_rows
        # fmt: off
        #$ omp end parallel
        # fmt: on
        offsets = np.zeros(num_threads, dtype=int)
        total_rows = 0
        for thread in range(num_threads):
            offsets[thread] = total_rows
            total_rows += num_rows[thread]
        buffer = np.zeros((max(total_rows, 1), num_y, num_components))
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
            offsets[thread],
            first_rows[thread],
            0,
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
        # fmt: on
        # the buffer rows that hold each x-row, in thread order (so the sum is
        # reproducible); a band past the last row wraps to the first ones
        starts = np.zeros((num_x, 2 * num_threads), dtype=int)
        num_starts = np.zeros(num_x, dtype=int)
        for thread in range(num_threads):
            for band_row in range(num_rows[thread]):
                i = (first_rows[thread] + band_row) % num_x
                starts[i, num_starts[i]] = offsets[thread] + band_row
                num_starts[i] += 1
        # fmt: off
        #$ omp parallel for private(j, component, total, index) schedule(dynamic, 4)
        # fmt: on
        for i in range(num_x):
            for j in range(num_y):
                for component in range(num_components):
                    total = 0.0
                    for index in range(num_starts[i]):
                        total += buffer[starts[i, index], j, component]
                    field[i, j, component] += total
