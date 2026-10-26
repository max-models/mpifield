"""Trilinear deposit of marker values onto a 3D grid, any number of components.

The Pyccel version (CPU) of the ``scatter_linear_3d`` kernel; ``scatter_linear_3d_cuda.cu`` next to
this file is the CUDA version (GPU), with the same arguments.  Both are loaded
by :mod:`mpifield.kernels.catalog`; this file is compiled with Pyccel there.

With OpenMP the markers are split over the threads in contiguous chunks, each
depositing into its own buffer, and the buffers are summed into ``field``: no
atomic updates, and the result is reproducible for a given number of threads (the
summation order, hence the last bits, differs from the serial one). A thread's
buffer covers only the x-planes its markers touch: markers sorted by cell
(e.g. picnic's ``Markers.sort_by_cell``) give each thread a narrow
band of planes, so zeroing and summing the buffers costs about one grid instead
of one grid per thread; unsorted markers give each thread the whole grid, as
before. On a periodic x axis the band runs past the last plane (``num_x`` is
plane 0 again) and is folded back in the sum. Few markers are deposited serially.

``positions`` is ``(3, N)`` and ``values`` ``(components, N)`` (component-major).
"""

import numpy as np
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

# Fewest markers worth the per-thread buffers.
MIN_PARALLEL_MARKERS = 20000


def _scatter_chunk_3d(
    positions: "float[:,:]",
    values: "float[:,:]",
    buffer: "float[:,:]",
    offset: int,
    first_plane: int,
    wrap: int,
    first: int,
    last: int,
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
    """Deposit the markers ``first`` to ``last`` into the rows ``offset`` onward of ``buffer``.

    Row ``offset`` of ``buffer`` is the first row of x-plane ``first_plane``.
    With ``wrap`` 1 the right x-node of a periodic axis wraps to plane 0 (the
    buffer is the whole grid); with 0 it is plane ``num_x`` of the band.
    """
    num_components = values.shape[0]
    # Multiplying by the inverse spacing, and clamping with comparisons: Pyccel
    # turns min/max into variadic C functions, a call each that the compiler
    # cannot inline.
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    inverse_z = 1.0 / cell_size_z
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
        # the cell and the weights along each axis; a periodic axis takes the
        # floor modulo only for a position outside [0, n) cells
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
        fx = rel_x - i_left
        if fx < 0.0:
            fx = 0.0
        if fx > 1.0:
            fx = 1.0
        rel_y = (positions[1, marker] - lower_y) * inverse_y
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
        fy = rel_y - j_left
        if fy < 0.0:
            fy = 0.0
        if fy > 1.0:
            fy = 1.0
        rel_z = (positions[2, marker] - lower_z) * inverse_z
        if periodic_z == 1:
            if not (rel_z >= 0.0 and rel_z < num_z):
                rel_z = rel_z - num_z * (rel_z // num_z)
            k_left = int(rel_z)
            if k_left >= num_z:
                k_left = num_z - 1
            k_right = k_left + 1
            if k_right == num_z:
                k_right = 0
        else:
            k_left = int(rel_z) if rel_z >= 0.0 else 0
            if k_left > num_z - 2:
                k_left = num_z - 2
            k_right = k_left + 1
        fz = rel_z - k_left
        if fz < 0.0:
            fz = 0.0
        if fz > 1.0:
            fz = 1.0
        # the x-planes in the buffer: a band runs past the last plane, the
        # whole grid (wrap) folds it back to plane 0
        i_right = i_left + 1
        if wrap == 1 and i_right == num_x:
            i_right = 0
        gx = 1.0 - fx
        gy = 1.0 - fy
        gz = 1.0 - fz
        w00 = gx * gy
        w01 = gx * fy
        w10 = fx * gy
        w11 = fx * fy
        r00 = offset + ((i_left - first_plane) * num_y + j_left) * num_z
        r01 = offset + ((i_left - first_plane) * num_y + j_right) * num_z
        r10 = offset + ((i_right - first_plane) * num_y + j_left) * num_z
        r11 = offset + ((i_right - first_plane) * num_y + j_right) * num_z
        if num_components == 1:
            # the charge deposit: one value, no loop over components
            buffer[r00 + k_left, 0] += w00 * gz * single
            buffer[r00 + k_right, 0] += w00 * fz * single
            buffer[r01 + k_left, 0] += w01 * gz * single
            buffer[r01 + k_right, 0] += w01 * fz * single
            buffer[r10 + k_left, 0] += w10 * gz * single
            buffer[r10 + k_right, 0] += w10 * fz * single
            buffer[r11 + k_left, 0] += w11 * gz * single
            buffer[r11 + k_right, 0] += w11 * fz * single
        else:
            for component in range(num_components):
                value = values[component, marker]
                buffer[r00 + k_left, component] += w00 * gz * value
                buffer[r00 + k_right, component] += w00 * fz * value
                buffer[r01 + k_left, component] += w01 * gz * value
                buffer[r01 + k_right, component] += w01 * fz * value
                buffer[r10 + k_left, component] += w10 * gz * value
                buffer[r10 + k_right, component] += w10 * fz * value
                buffer[r11 + k_left, component] += w11 * gz * value
                buffer[r11 + k_right, component] += w11 * fz * value


def _x_planes(
    positions: "float[:,:]",
    first: int,
    last: int,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> "tuple[int, int]":
    """The first x-plane and the number of planes the markers ``first`` to ``last`` touch.

    The band covers the left x-node of every marker (also the zero-value ones
    that deposit nothing, so the scan reads only ``positions[0]``) and the node
    after it, without the periodic wrap: up to plane ``num_x``. ``(0, 0)`` for
    no markers.
    """
    lowest = num_x
    highest = -1
    inverse_x = 1.0 / cell_size_x
    for marker in range(first, last):
        # the left x-node of _scatter_chunk_3d, compared without min/max
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


def scatter_linear_3d(
    positions: "float[:,:]",
    values: "float[:,:]",
    field: "float[:,:]",
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
    """Deposit ``values`` (components, N) onto ``field``.

    ``field`` is the grid ``(nx, ny, nz, components)`` reshaped to
    ``(nx * ny * nz, components)``: cell ``(i, j, k)`` is row ``(i * ny + j) * nz + k``.
    """
    num_markers = positions.shape[1]
    num_rows = field.shape[0]
    num_components = field.shape[1]
    num_threads = int(omp_get_max_threads())
    if num_threads == 1 or num_markers < MIN_PARALLEL_MARKERS or num_markers < num_rows:
        _scatter_chunk_3d(
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
            lower_z,
            cell_size_x,
            cell_size_y,
            cell_size_z,
            num_x,
            num_y,
            num_z,
            periodic_x,
            periodic_y,
            periodic_z,
        )
    else:
        plane = num_y * num_z
        chunk = (num_markers + num_threads - 1) // num_threads
        first_planes = np.zeros(num_threads, dtype=int)
        num_planes = np.zeros(num_threads, dtype=int)
        # fmt: off
        #$ omp parallel num_threads(num_threads) private(thread, first, last, band_first, band_planes)
        # fmt: on
        thread = int(omp_get_thread_num())
        first = min(thread * chunk, num_markers)
        last = min(first + chunk, num_markers)
        band_first, band_planes = _x_planes(
            positions, first, last, lower_x, cell_size_x, num_x, periodic_x
        )
        first_planes[thread] = band_first
        num_planes[thread] = band_planes
        # fmt: off
        #$ omp end parallel
        # fmt: on
        offsets = np.zeros(num_threads, dtype=int)
        total_planes = 0
        for thread in range(num_threads):
            offsets[thread] = total_planes * plane
            total_planes += num_planes[thread]
        buffer = np.zeros((max(total_planes, 1) * plane, num_components))
        # fmt: off
        #$ omp parallel num_threads(num_threads) private(thread, first, last)
        # fmt: on
        thread = int(omp_get_thread_num())
        first = min(thread * chunk, num_markers)
        last = min(first + chunk, num_markers)
        _scatter_chunk_3d(
            positions,
            values,
            buffer,
            offsets[thread],
            first_planes[thread],
            0,
            first,
            last,
            lower_x,
            lower_y,
            lower_z,
            cell_size_x,
            cell_size_y,
            cell_size_z,
            num_x,
            num_y,
            num_z,
            periodic_x,
            periodic_y,
            periodic_z,
        )
        # fmt: off
        #$ omp end parallel
        # fmt: on
        # the buffer rows that hold each x-plane, in thread order (so the sum is
        # reproducible); a band past the last plane wraps to the first ones
        starts = np.zeros((num_x, 2 * num_threads), dtype=int)
        num_starts = np.zeros(num_x, dtype=int)
        for thread in range(num_threads):
            for band_plane in range(num_planes[thread]):
                x_plane = (first_planes[thread] + band_plane) % num_x
                starts[x_plane, num_starts[x_plane]] = (
                    offsets[thread] + band_plane * plane
                )
                num_starts[x_plane] += 1
        # fmt: off
        #$ omp parallel for private(row, component, total, index) schedule(dynamic, 1)
        # fmt: on
        for x_plane in range(num_x):
            for row in range(plane):
                for component in range(num_components):
                    total = 0.0
                    for index in range(num_starts[x_plane]):
                        total += buffer[starts[x_plane, index] + row, component]
                    field[x_plane * plane + row, component] += total
