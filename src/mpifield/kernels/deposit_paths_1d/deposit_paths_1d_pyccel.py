"""The current of charges moving along straight paths, on the edges of a 1D grid.

The Pyccel version (CPU) of the ``deposit_paths_1d`` kernel;
``deposit_paths_1d_cuda.cu`` next to this file is the CUDA version (GPU) and
``deposit_paths_1d_numpy.py`` the NumPy version, with the same arguments. Loaded by
`mpifield.kernels.catalog`; this file is compiled with Pyccel there.

``edges`` ``(nx, 1)`` holds the line integrals along the cells (``[i, 0]``, from node
``i`` to ``i + 1``). Every charge goes from ``start`` to ``end`` (``(1, N)``, not
wrapped on a periodic axis); its path is cut at the nodes, and each piece adds ``q``
times its length (in cells) to the edge of its cell. On a bounded axis the path is
clamped to the grid first, as `LINEAR` clamps a position, so ``G^T j`` is `LINEAR`'s
deposit at the ends minus at the starts. A zero or non-finite path deposits nothing.

With OpenMP the paths are split over the threads in contiguous chunks, each
depositing into its own copy of the edges, summed in thread order. Few paths are
deposited on one thread.
"""

from math import ceil, floor

import numpy as np
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

from mpifield.kernels.linear_cell_pyccel import left_node, periodic_position
from mpifield.kernels.whitney_cell_pyccel import clamp_to_axis, plane_time

# Fewest paths worth the per-thread copies of the edges.
MIN_PARALLEL_PATHS = 20000


def _deposit_chunk_1d(
    start: "float[:,:]",
    end: "float[:,:]",
    charges: "float[:]",
    buffer: "float[:,:,:]",
    slot: int,
    first: int,
    last: int,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> None:
    """Deposit the paths ``first`` to ``last`` into ``buffer[slot]``."""
    inverse_x = 1.0 / cell_size_x
    for path in range(first, last):
        q = charges[path]
        if q == 0.0:
            continue
        ax = (start[0, path] - lower_x) * inverse_x
        bx = (end[0, path] - lower_x) * inverse_x
        if not (ax > -1.0e300 and ax < 1.0e300 and bx > -1.0e300 and bx < 1.0e300):
            continue
        dx = bx - ax
        step_x = 1.0
        plane_x = floor(ax) + 1.0
        if dx < 0.0:
            step_x = -1.0
            plane_x = ceil(ax) - 1.0
        time = 0.0
        x0 = clamp_to_axis(ax, num_x, periodic_x)
        while time < 1.0:
            following = plane_time(plane_x, ax, dx)
            if following < 1.0:
                x1 = clamp_to_axis(ax + following * dx, num_x, periodic_x)
                plane_x += step_x
            else:
                following = 1.0
                x1 = clamp_to_axis(bx, num_x, periodic_x)
            mid_x = 0.5 * (x0 + x1)
            if periodic_x == 1:
                mid_x = periodic_position(mid_x, num_x)
            i_left = left_node(mid_x, num_x, periodic_x)
            buffer[slot, i_left, 0] += q * (x1 - x0)
            time = following
            x0 = x1


def deposit_paths_1d(
    start: "float[:,:]",
    end: "float[:,:]",
    charges: "float[:]",
    edges: "float[:,:]",
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> None:
    """Add the current of the paths from ``start`` to ``end`` (1, N) to ``edges`` (nx, 1)."""
    num_paths = charges.shape[0]
    num_threads = int(omp_get_max_threads())
    if num_paths < MIN_PARALLEL_PATHS or num_paths < edges.shape[0]:
        num_threads = 1
    # one copy of the edges per thread, the thread's slot passed with the whole
    # buffer: a slice ``buffer[thread]`` would be one view shared by the threads
    buffer = np.zeros((num_threads, edges.shape[0], 1))
    chunk = (num_paths + num_threads - 1) // num_threads
    # fmt: off
    #$ omp parallel num_threads(num_threads) private(thread, first, last)
    # fmt: on
    thread = int(omp_get_thread_num())
    first = min(thread * chunk, num_paths)
    last = min(first + chunk, num_paths)
    _deposit_chunk_1d(
        start, end, charges, buffer, thread, first, last,
        lower_x, cell_size_x,
        num_x, periodic_x,
    )  # fmt: skip
    # fmt: off
    #$ omp end parallel
    #$ omp parallel for private(a, thread)
    # fmt: on
    for i in range(edges.shape[0]):
        for a in range(1):
            for thread in range(num_threads):
                edges[i, a] += buffer[thread, i, a]
