"""B-spline deposit of degree 0 to 3 of marker values onto a 3D grid, any number of components.

The Pyccel version (CPU) of the ``scatter_spline_3d`` kernel; ``scatter_spline_3d_cuda.cu`` next to
this file is the CUDA version (GPU), with the same arguments.  Both are loaded
by `mpifield.kernels.catalog`; this file is compiled with Pyccel there.

Each marker deposits onto the ``degree + 1`` nodes of its stencil along each axis
(``spline_cell_pyccel.spline_cell``). With OpenMP the markers are split over the
threads in contiguous chunks, each depositing into its own copy of the grid, and the
copies are summed into ``field`` in thread order: no atomic updates, and the result is
reproducible for a given number of threads. Few markers, or copies that would together
hold more than ``MAX_BUFFER_VALUES`` values, are deposited serially.

``positions`` is ``(3, N)``, ``values`` ``(components, N)`` and ``field``
``(nx, ny, nz, components)``; ``degree`` is 0 to 3.
"""

import numpy as np
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

from mpifield.kernels.spline_cell_pyccel import spline_cell

# Fewest markers worth the per-thread grid copies.
MIN_PARALLEL_MARKERS = 20000
# Most values of all the per-thread grid copies together (128 MB).
MAX_BUFFER_VALUES = 16777216


def _scatter_chunk_3d(
    positions: "float[:,:]",
    values: "float[:,:]",
    buffer: "float[:,:,:,:]",
    offset: int,
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
    degree: int,
) -> None:
    """Deposit the markers ``first`` to ``last`` into the x-rows ``offset`` onward of ``buffer``."""
    num_components = values.shape[0]
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    inverse_z = 1.0 / cell_size_z
    node_x = np.zeros(4, dtype=int)
    weight_x = np.zeros(4)
    node_y = np.zeros(4, dtype=int)
    weight_y = np.zeros(4)
    node_z = np.zeros(4, dtype=int)
    weight_z = np.zeros(4)
    size = degree + 1
    for marker in range(first, last):
        # a zero-weight marker (a dead one left in storage) deposits nothing,
        # whatever its position
        deposit = False
        for component in range(num_components):
            if values[component, marker] != 0.0:
                deposit = True
        if not deposit:
            continue
        spline_cell(
            (positions[0, marker] - lower_x) * inverse_x,
            num_x,
            periodic_x,
            degree,
            node_x,
            weight_x,
        )
        spline_cell(
            (positions[1, marker] - lower_y) * inverse_y,
            num_y,
            periodic_y,
            degree,
            node_y,
            weight_y,
        )
        spline_cell(
            (positions[2, marker] - lower_z) * inverse_z,
            num_z,
            periodic_z,
            degree,
            node_z,
            weight_z,
        )
        for a in range(size):
            row = offset + node_x[a]
            for b in range(size):
                weight_ab = weight_x[a] * weight_y[b]
                for e in range(size):
                    weight = weight_ab * weight_z[e]
                    for component in range(num_components):
                        buffer[row, node_y[b], node_z[e], component] += (
                            weight * values[component, marker]
                        )


def scatter_spline_3d(
    positions: "float[:,:]",
    values: "float[:,:]",
    field: "float[:,:,:,:]",
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
    degree: int,
) -> None:
    """Deposit ``values`` (components, N) onto ``field`` (nx, ny, nz, components)."""
    num_markers = positions.shape[1]
    num_components = values.shape[0]
    num_threads = int(omp_get_max_threads())
    grid_values = field.shape[0] * field.shape[1] * field.shape[2] * num_components
    if (
        num_threads == 1
        or num_markers < MIN_PARALLEL_MARKERS
        or num_markers < grid_values
        or num_threads * grid_values > MAX_BUFFER_VALUES
    ):
        _scatter_chunk_3d(
            positions,
            values,
            field,
            0,
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
            degree,
        )
    else:
        buffer = np.zeros(
            (
                num_threads * field.shape[0],
                field.shape[1],
                field.shape[2],
                num_components,
            )
        )
        chunk = (num_markers + num_threads - 1) // num_threads
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
            thread * field.shape[0],
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
            degree,
        )
        # fmt: off
        #$ omp end parallel
        #$ omp parallel for private(j, k, component, total, thread)
        # fmt: on
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                for k in range(field.shape[2]):
                    for component in range(num_components):
                        total = 0.0
                        for thread in range(num_threads):
                            total += buffer[
                                thread * field.shape[0] + i, j, k, component
                            ]
                        field[i, j, k, component] += total
