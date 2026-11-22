"""B-spline interpolation of degree 0 to 3 of a 2D grid field at the markers.

The Pyccel version (CPU) of the ``gather_spline_2d`` kernel; ``gather_spline_2d_cuda.cu`` next to
this file is the CUDA version (GPU), with the same arguments.  Both are loaded
by `mpifield.kernels.catalog`; this file is compiled with Pyccel there.

The transpose of ``scatter_spline_2d``: each marker reads the ``degree + 1`` nodes of
its stencil along each axis (``spline_cell_pyccel.spline_cell``). With OpenMP the
markers are split over the threads in contiguous chunks.

``positions`` is ``(2, N)``, ``field`` ``(nx, ny, components)`` and ``result``
``(components, N)``; ``degree`` is 0 to 3.
"""

import numpy as np
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

from mpifield.kernels.spline_cell_pyccel import spline_cell

# Fewest markers worth starting the threads for.
MIN_PARALLEL_MARKERS = 4096


def _gather_chunk_2d(
    positions: "float[:,:]",
    field: "float[:,:,:]",
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
    """Interpolate ``field`` at the markers ``first`` to ``last`` into ``result``."""
    num_components = field.shape[2]
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    node_x = np.zeros(4, dtype=int)
    weight_x = np.zeros(4)
    node_y = np.zeros(4, dtype=int)
    weight_y = np.zeros(4)
    size = degree + 1
    for marker in range(first, last):
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
        for component in range(num_components):
            total = 0.0
            for a in range(size):
                for b in range(size):
                    total += (
                        weight_x[a]
                        * weight_y[b]
                        * field[node_x[a], node_y[b], component]
                    )
            result[component, marker] = total


def gather_spline_2d(
    positions: "float[:,:]",
    field: "float[:,:,:]",
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
    """Interpolate ``field`` (nx, ny, components) at the markers into ``result`` (components, N)."""
    num_markers = positions.shape[1]
    num_threads = int(omp_get_max_threads())
    if num_threads == 1 or num_markers < MIN_PARALLEL_MARKERS:
        _gather_chunk_2d(
            positions,
            field,
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
            field,
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
