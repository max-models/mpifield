"""The gather of a k-form of B-spline degree 1 to 3 at the markers, in 1D.

The Pyccel version (CPU) of the ``gather_forms_spline_1d`` kernel; ``gather_forms_spline_1d_cuda.cu``
next to this file is the CUDA version (GPU), with the same arguments. Component ``c`` of
the ``form``-form is the tensor product of ``B_p`` along the axes it is not staggered on
and ``B_(p-1)`` shifted by half a cell along those it is (``spline_forms.py``); the result
is ``sum W f`` (no division by the measures). With OpenMP the markers are split over the
threads in contiguous chunks.

``positions`` is ``(1, N)``, ``field`` ``(nx, components)`` and ``result`` ``(components, N)``.
"""

import numpy as np
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

from mpifield.kernels.spline_cell_pyccel import spline_cell
from mpifield.kernels.spline_edges_pyccel import along_stencil, staggered

# Fewest markers worth starting the threads for.
MIN_PARALLEL_MARKERS = 4096


def _gather_chunk_1d(
    positions: "float[:,:]",
    field: "float[:,:]",
    result: "float[:,:]",
    first: int,
    last: int,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
    degree: int,
    form: int,
) -> None:
    """The gather at the markers ``first`` to ``last``."""
    num_components = field.shape[1]
    inverse_x = 1.0 / cell_size_x
    index_x = np.zeros(4, dtype=int)
    weight_x = np.zeros(4)
    exists_x = np.zeros(4, dtype=int)
    for marker in range(first, last):
        rel_x = (positions[0, marker] - lower_x) * inverse_x
        if periodic_x == 0:
            if rel_x < 0.0:
                rel_x = 0.0
            if rel_x > num_x - 1:
                rel_x = num_x - 1.0
        for component in range(num_components):
            if staggered(form, 1, component, 0) == 1:
                along_stencil(
                    rel_x, num_x, periodic_x, degree, index_x, weight_x, exists_x
                )
                size_x = degree
            else:
                spline_cell(rel_x, num_x, periodic_x, degree, index_x, weight_x)
                for i in range(degree + 1):
                    exists_x[i] = 1
                size_x = degree + 1
            total = 0.0
            for a in range(size_x):
                if exists_x[a] == 1:
                    total += weight_x[a] * field[index_x[a], component]
            result[component, marker] = total


def gather_forms_spline_1d(
    positions: "float[:,:]",
    field: "float[:,:]",
    result: "float[:,:]",
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
    degree: int,
    form: int,
) -> None:
    """Write ``sum W f`` of ``field`` at ``positions`` into ``result``."""
    num_markers = positions.shape[1]
    num_threads = int(omp_get_max_threads())
    if num_threads == 1 or num_markers < MIN_PARALLEL_MARKERS:
        _gather_chunk_1d(
            positions,
            field,
            result,
            0,
            num_markers,
            lower_x,
            cell_size_x,
            num_x,
            periodic_x,
            degree,
            form,
        )
    else:
        chunk = (num_markers + num_threads - 1) // num_threads
        # fmt: off
        #$ omp parallel num_threads(num_threads) private(thread, first, last)
        # fmt: on
        thread = int(omp_get_thread_num())
        first = min(thread * chunk, num_markers)
        last = min(first + chunk, num_markers)
        _gather_chunk_1d(
            positions,
            field,
            result,
            first,
            last,
            lower_x,
            cell_size_x,
            num_x,
            periodic_x,
            degree,
            form,
        )
        # fmt: off
        #$ omp end parallel
        # fmt: on
