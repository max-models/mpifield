"""Quadratic B-spline interpolation of a 3D grid field at the markers.

The Pyccel version (CPU) of the ``gather_quadratic_3d`` kernel; ``gather_quadratic_3d_cuda.cu`` next to
this file is the CUDA version (GPU), with the same arguments.  Both are loaded
by :mod:`mpifield.kernels.catalog`; this file is compiled with Pyccel there.

The transpose of ``scatter_quadratic_3d``: each marker reads its nearest node and
the two around it along each axis (``quadratic_cell.cuh``). With OpenMP the
markers are split over the threads in contiguous chunks.

``positions`` is ``(3, N)``, ``field`` ``(nx, ny, nz, components)`` and ``result``
``(components, N)``.
"""

import numpy as np
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

# Fewest markers worth starting the threads for.
MIN_PARALLEL_MARKERS = 4096


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


def _gather_chunk_3d(
    positions: "float[:,:]",
    field: "float[:,:,:,:]",
    result: "float[:,:]",
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
    """Interpolate ``field`` at the markers ``first`` to ``last`` into ``result``."""
    num_components = field.shape[3]
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    inverse_z = 1.0 / cell_size_z
    node_x = np.zeros(3, dtype=int)
    weight_x = np.zeros(3)
    node_y = np.zeros(3, dtype=int)
    weight_y = np.zeros(3)
    node_z = np.zeros(3, dtype=int)
    weight_z = np.zeros(3)
    for marker in range(first, last):
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
        _quadratic_cell(
            (positions[2, marker] - lower_z) * inverse_z,
            num_z,
            periodic_z,
            node_z,
            weight_z,
        )
        for component in range(num_components):
            total = 0.0
            for a in range(3):
                for b in range(3):
                    weight_ab = weight_x[a] * weight_y[b]
                    for e in range(3):
                        total += (
                            weight_ab
                            * weight_z[e]
                            * field[node_x[a], node_y[b], node_z[e], component]
                        )
            result[component, marker] = total


def gather_quadratic_3d(
    positions: "float[:,:]",
    field: "float[:,:,:,:]",
    result: "float[:,:]",
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
    """Interpolate ``field`` (nx, ny, nz, components) at the markers into ``result`` (components, N)."""
    num_markers = positions.shape[1]
    num_threads = int(omp_get_max_threads())
    if num_threads == 1 or num_markers < MIN_PARALLEL_MARKERS:
        _gather_chunk_3d(
            positions,
            field,
            result,
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
        chunk = (num_markers + num_threads - 1) // num_threads
        # fmt: off
        #$ omp parallel num_threads(num_threads) private(thread, first, last)
        # fmt: on
        thread = int(omp_get_thread_num())
        first = min(thread * chunk, num_markers)
        last = min(first + chunk, num_markers)
        _gather_chunk_3d(
            positions,
            field,
            result,
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
