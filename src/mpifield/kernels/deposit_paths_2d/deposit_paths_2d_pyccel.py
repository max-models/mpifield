"""The current of charges moving along straight paths, on the edges of a 2D grid.

The Pyccel version (CPU) of the ``deposit_paths_2d`` kernel;
``deposit_paths_2d_cuda.cu`` next to this file is the CUDA version (GPU) and
``deposit_paths_2d_numpy.py`` the NumPy version, with the same arguments. Loaded by
`mpifield.kernels.catalog`; this file is compiled with Pyccel there.

``edges`` ``(nx, ny, 2)`` holds the line integrals along the x-edges (``[..., 0]``,
from node ``(i, j)`` to ``(i + 1, j)``) and the y-edges (``[..., 1]``). Every charge
goes from ``start`` to ``end`` (``(2, N)``, not wrapped on a periodic axis); its path
is cut where it crosses a grid line, and each piece adds ``q`` times its length along
an axis (in cells) to the two edges of its cell along that axis, weighted linearly
by the piece's midpoint across the cell. On a bounded axis the path is clamped to
the grid first, as `LINEAR` clamps a position, so the current is that of `LINEAR`'s
charge: ``G^T j`` is its deposit at the ends minus at the starts. A zero or
non-finite path deposits nothing.

With OpenMP the paths are split over the threads in contiguous chunks, each
depositing into its own copy of the edges, summed in thread order. Few paths are
deposited serially.
"""

from math import ceil, floor

import numpy as np
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

from mpifield.kernels.linear_cell_pyccel import (
    cell_fraction,
    left_node,
    periodic_position,
    right_node,
)
from mpifield.kernels.whitney_cell_pyccel import (
    clamp_to_axis,
    edge_weight,
    plane_time,
)

# Fewest paths worth the per-thread copies of the edges.
MIN_PARALLEL_PATHS = 20000


def _deposit_chunk_2d(
    start: "float[:,:]",
    end: "float[:,:]",
    charges: "float[:]",
    edges: "float[:,:,:]",
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
    """Deposit the paths ``first`` to ``last`` into ``edges``."""
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    for path in range(first, last):
        q = charges[path]
        if q == 0.0:
            continue
        ax = (start[0, path] - lower_x) * inverse_x
        ay = (start[1, path] - lower_y) * inverse_y
        bx = (end[0, path] - lower_x) * inverse_x
        by = (end[1, path] - lower_y) * inverse_y
        if not (
            ax > -1.0e300
            and ax < 1.0e300
            and ay > -1.0e300
            and ay < 1.0e300
            and bx > -1.0e300
            and bx < 1.0e300
            and by > -1.0e300
            and by < 1.0e300
        ):
            continue
        dx = bx - ax
        dy = by - ay
        # the next grid line each coordinate reaches, and the step to the one after
        step_x = 1.0
        plane_x = floor(ax) + 1.0
        if dx < 0.0:
            step_x = -1.0
            plane_x = ceil(ax) - 1.0
        step_y = 1.0
        plane_y = floor(ay) + 1.0
        if dy < 0.0:
            step_y = -1.0
            plane_y = ceil(ay) - 1.0
        time = 0.0
        x0 = clamp_to_axis(ax, num_x, periodic_x)
        y0 = clamp_to_axis(ay, num_y, periodic_y)
        while time < 1.0:
            time_x = plane_time(plane_x, ax, dx)
            time_y = plane_time(plane_y, ay, dy)
            following = 1.0
            if time_x < following:
                following = time_x
            if time_y < following:
                following = time_y
            if following < 1.0:
                x1 = clamp_to_axis(ax + following * dx, num_x, periodic_x)
                y1 = clamp_to_axis(ay + following * dy, num_y, periodic_y)
            else:
                x1 = clamp_to_axis(bx, num_x, periodic_x)
                y1 = clamp_to_axis(by, num_y, periodic_y)
            if time_x <= following:
                plane_x += step_x
            if time_y <= following:
                plane_y += step_y
            # the piece's cell, from its midpoint, and the midpoint across it
            mid_x = 0.5 * (x0 + x1)
            mid_y = 0.5 * (y0 + y1)
            if periodic_x == 1:
                mid_x = periodic_position(mid_x, num_x)
            if periodic_y == 1:
                mid_y = periodic_position(mid_y, num_y)
            i_left = left_node(mid_x, num_x, periodic_x)
            i_right = right_node(i_left, num_x, periodic_x)
            j_left = left_node(mid_y, num_y, periodic_y)
            j_right = right_node(j_left, num_y, periodic_y)
            fx = cell_fraction(mid_x, i_left)
            fy = cell_fraction(mid_y, j_left)
            along_x = q * (x1 - x0)
            along_y = q * (y1 - y0)
            edges[i_left, j_left, 0] += edge_weight(fy, 0) * along_x
            edges[i_left, j_right, 0] += edge_weight(fy, 1) * along_x
            edges[i_left, j_left, 1] += edge_weight(fx, 0) * along_y
            edges[i_right, j_left, 1] += edge_weight(fx, 1) * along_y
            time = following
            x0 = x1
            y0 = y1


def deposit_paths_2d(
    start: "float[:,:]",
    end: "float[:,:]",
    charges: "float[:]",
    edges: "float[:,:,:]",
    lower_x: float,
    lower_y: float,
    cell_size_x: float,
    cell_size_y: float,
    num_x: int,
    num_y: int,
    periodic_x: int,
    periodic_y: int,
) -> None:
    """Add the current of the paths from ``start`` to ``end`` (2, N) to ``edges`` (nx, ny, 2)."""
    num_paths = charges.shape[0]
    num_threads = int(omp_get_max_threads())
    if (
        num_threads == 1
        or num_paths < MIN_PARALLEL_PATHS
        or num_paths < edges.shape[0] * edges.shape[1]
    ):
        _deposit_chunk_2d(
            start, end, charges, edges, 0, num_paths,
            lower_x, lower_y, cell_size_x, cell_size_y,
            num_x, num_y, periodic_x, periodic_y,
        )  # fmt: skip
        return
    buffer = np.zeros((num_threads, edges.shape[0], edges.shape[1], 2))
    chunk = (num_paths + num_threads - 1) // num_threads
    # fmt: off
    #$ omp parallel num_threads(num_threads) private(thread, first, last)
    # fmt: on
    thread = int(omp_get_thread_num())
    first = min(thread * chunk, num_paths)
    last = min(first + chunk, num_paths)
    _deposit_chunk_2d(
        start, end, charges, buffer[thread], first, last,
        lower_x, lower_y, cell_size_x, cell_size_y,
        num_x, num_y, periodic_x, periodic_y,
    )  # fmt: skip
    # fmt: off
    #$ omp end parallel
    #$ omp parallel for private(j, a, thread)
    # fmt: on
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            for a in range(2):
                for thread in range(num_threads):
                    edges[i, j, a] += buffer[thread, i, j, a]
