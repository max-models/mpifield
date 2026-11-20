"""The current of charges moving along straight paths, on the edges of a 3D grid.

The Pyccel version (CPU) of the ``deposit_paths_3d`` kernel;
``deposit_paths_3d_cuda.cu`` next to this file is the CUDA version (GPU) and
``deposit_paths_3d_numpy.py`` the NumPy version, with the same arguments. Loaded by
`mpifield.kernels.catalog`; this file is compiled with Pyccel there.

``edges`` ``(nx, ny, nz, 3)`` holds the line integrals along the x-, y- and z-edges
(``[..., a]``, from node ``(i, j, k)`` to its neighbour along axis ``a``). Every charge
goes from ``start`` to ``end`` (``(3, N)``, not wrapped on a periodic axis); its path
is cut where it crosses a grid plane, and each piece adds ``q`` times its length
along an axis (in cells) to the four edges of its cell along that axis, times the
average of their bilinear weights over the piece (`edge_path_weight`). On a bounded
axis the path is clamped to the grid first, as `LINEAR` clamps a position, so
``G^T j`` is `LINEAR`'s deposit at the ends minus at the starts. A zero or
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
    edge_path_weight,
    plane_time,
)

# Fewest paths worth the per-thread copies of the edges.
MIN_PARALLEL_PATHS = 20000


def _deposit_chunk_3d(
    start: "float[:,:]",
    end: "float[:,:]",
    charges: "float[:]",
    edges: "float[:,:,:,:]",
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
    """Deposit the paths ``first`` to ``last`` into ``edges``."""
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    inverse_z = 1.0 / cell_size_z
    for path in range(first, last):
        q = charges[path]
        if q == 0.0:
            continue
        ax = (start[0, path] - lower_x) * inverse_x
        ay = (start[1, path] - lower_y) * inverse_y
        az = (start[2, path] - lower_z) * inverse_z
        bx = (end[0, path] - lower_x) * inverse_x
        by = (end[1, path] - lower_y) * inverse_y
        bz = (end[2, path] - lower_z) * inverse_z
        if not (
            ax > -1.0e300
            and ax < 1.0e300
            and ay > -1.0e300
            and ay < 1.0e300
            and az > -1.0e300
            and az < 1.0e300
            and bx > -1.0e300
            and bx < 1.0e300
            and by > -1.0e300
            and by < 1.0e300
            and bz > -1.0e300
            and bz < 1.0e300
        ):
            continue
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        # the next grid plane each coordinate reaches, and the step to the one after
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
        step_z = 1.0
        plane_z = floor(az) + 1.0
        if dz < 0.0:
            step_z = -1.0
            plane_z = ceil(az) - 1.0
        time = 0.0
        x0 = clamp_to_axis(ax, num_x, periodic_x)
        y0 = clamp_to_axis(ay, num_y, periodic_y)
        z0 = clamp_to_axis(az, num_z, periodic_z)
        while time < 1.0:
            time_x = plane_time(plane_x, ax, dx)
            time_y = plane_time(plane_y, ay, dy)
            time_z = plane_time(plane_z, az, dz)
            following = 1.0
            if time_x < following:
                following = time_x
            if time_y < following:
                following = time_y
            if time_z < following:
                following = time_z
            if following < 1.0:
                x1 = clamp_to_axis(ax + following * dx, num_x, periodic_x)
                y1 = clamp_to_axis(ay + following * dy, num_y, periodic_y)
                z1 = clamp_to_axis(az + following * dz, num_z, periodic_z)
            else:
                x1 = clamp_to_axis(bx, num_x, periodic_x)
                y1 = clamp_to_axis(by, num_y, periodic_y)
                z1 = clamp_to_axis(bz, num_z, periodic_z)
            if time_x <= following:
                plane_x += step_x
            if time_y <= following:
                plane_y += step_y
            if time_z <= following:
                plane_z += step_z
            # the piece's cell, from its midpoint, and the midpoint across it
            mid_x = 0.5 * (x0 + x1)
            mid_y = 0.5 * (y0 + y1)
            mid_z = 0.5 * (z0 + z1)
            if periodic_x == 1:
                mid_x = periodic_position(mid_x, num_x)
            if periodic_y == 1:
                mid_y = periodic_position(mid_y, num_y)
            if periodic_z == 1:
                mid_z = periodic_position(mid_z, num_z)
            i0 = left_node(mid_x, num_x, periodic_x)
            i1 = right_node(i0, num_x, periodic_x)
            j0 = left_node(mid_y, num_y, periodic_y)
            j1 = right_node(j0, num_y, periodic_y)
            k0 = left_node(mid_z, num_z, periodic_z)
            k1 = right_node(k0, num_z, periodic_z)
            fx = cell_fraction(mid_x, i0)
            fy = cell_fraction(mid_y, j0)
            fz = cell_fraction(mid_z, k0)
            delta_x = x1 - x0
            delta_y = y1 - y0
            delta_z = z1 - z0
            along_x = q * delta_x
            along_y = q * delta_y
            along_z = q * delta_z
            # the x-edges across y and z, the y-edges across x and z, the z-edges
            # across x and y
            edges[i0, j0, k0, 0] += along_x * edge_path_weight(
                fy, fz, delta_y, delta_z, 0, 0
            )
            edges[i0, j1, k0, 0] += along_x * edge_path_weight(
                fy, fz, delta_y, delta_z, 1, 0
            )
            edges[i0, j0, k1, 0] += along_x * edge_path_weight(
                fy, fz, delta_y, delta_z, 0, 1
            )
            edges[i0, j1, k1, 0] += along_x * edge_path_weight(
                fy, fz, delta_y, delta_z, 1, 1
            )
            edges[i0, j0, k0, 1] += along_y * edge_path_weight(
                fx, fz, delta_x, delta_z, 0, 0
            )
            edges[i1, j0, k0, 1] += along_y * edge_path_weight(
                fx, fz, delta_x, delta_z, 1, 0
            )
            edges[i0, j0, k1, 1] += along_y * edge_path_weight(
                fx, fz, delta_x, delta_z, 0, 1
            )
            edges[i1, j0, k1, 1] += along_y * edge_path_weight(
                fx, fz, delta_x, delta_z, 1, 1
            )
            edges[i0, j0, k0, 2] += along_z * edge_path_weight(
                fx, fy, delta_x, delta_y, 0, 0
            )
            edges[i1, j0, k0, 2] += along_z * edge_path_weight(
                fx, fy, delta_x, delta_y, 1, 0
            )
            edges[i0, j1, k0, 2] += along_z * edge_path_weight(
                fx, fy, delta_x, delta_y, 0, 1
            )
            edges[i1, j1, k0, 2] += along_z * edge_path_weight(
                fx, fy, delta_x, delta_y, 1, 1
            )
            time = following
            x0 = x1
            y0 = y1
            z0 = z1


def deposit_paths_3d(
    start: "float[:,:]",
    end: "float[:,:]",
    charges: "float[:]",
    edges: "float[:,:,:,:]",
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
    """Add the current of the paths from ``start`` to ``end`` (3, N) to ``edges`` (nx, ny, nz, 3)."""
    num_paths = charges.shape[0]
    num_threads = int(omp_get_max_threads())
    if (
        num_threads == 1
        or num_paths < MIN_PARALLEL_PATHS
        or num_paths < edges.shape[0] * edges.shape[1] * edges.shape[2]
    ):
        _deposit_chunk_3d(
            start, end, charges, edges, 0, num_paths,
            lower_x, lower_y, lower_z, cell_size_x, cell_size_y, cell_size_z,
            num_x, num_y, num_z, periodic_x, periodic_y, periodic_z,
        )  # fmt: skip
        return
    buffer = np.zeros((num_threads, edges.shape[0], edges.shape[1], edges.shape[2], 3))
    chunk = (num_paths + num_threads - 1) // num_threads
    # fmt: off
    #$ omp parallel num_threads(num_threads) private(thread, first, last)
    # fmt: on
    thread = int(omp_get_thread_num())
    first = min(thread * chunk, num_paths)
    last = min(first + chunk, num_paths)
    _deposit_chunk_3d(
        start, end, charges, buffer[thread], first, last,
        lower_x, lower_y, lower_z, cell_size_x, cell_size_y, cell_size_z,
        num_x, num_y, num_z, periodic_x, periodic_y, periodic_z,
    )  # fmt: skip
    # fmt: off
    #$ omp end parallel
    #$ omp parallel for private(j, k, a, thread)
    # fmt: on
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            for k in range(edges.shape[2]):
                for a in range(3):
                    for thread in range(num_threads):
                        edges[i, j, k, a] += buffer[thread, i, j, k, a]
