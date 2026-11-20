"""The edge (Whitney) field of a 3D grid at the markers.

The Pyccel version (CPU) of the ``gather_edges_3d`` kernel; ``gather_edges_3d_cuda.cu``
next to this file is the CUDA version (GPU) and ``gather_edges_3d_numpy.py`` the NumPy
version, with the same arguments. Loaded by `mpifield.kernels.catalog`; this file is
compiled with Pyccel there.

``edges`` ``(nx, ny, nz, 3)`` holds the line integrals along the x-, y- and z-edges.
In the cell of a marker (`LINEAR`'s cell: clamped on a bounded axis, wrapped on a
periodic one) the component along an axis is bilinear across the cell between its
four edges along that axis, divided by the cell size along them. ``positions`` and
``result`` are ``(3, N)``.
"""

from mpifield.kernels.linear_cell_pyccel import (
    cell_fraction,
    left_node,
    periodic_position,
    right_node,
)
from mpifield.kernels.whitney_cell_pyccel import edge_weight


def gather_edges_3d(
    positions: "float[:,:]",
    edges: "float[:,:,:,:]",
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
    """Write the field of ``edges`` at ``positions`` into ``result``."""
    num_markers = positions.shape[1]
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    inverse_z = 1.0 / cell_size_z
    # fmt: off
    #$ omp parallel for private(rel_x, rel_y, rel_z, i0, i1, j0, j1, k0, k1, fx, fy, fz) schedule(dynamic, 4096)
    # fmt: on
    for marker in range(num_markers):
        rel_x = (positions[0, marker] - lower_x) * inverse_x
        if periodic_x == 1:
            rel_x = periodic_position(rel_x, num_x)
        i0 = left_node(rel_x, num_x, periodic_x)
        i1 = right_node(i0, num_x, periodic_x)
        fx = cell_fraction(rel_x, i0)
        rel_y = (positions[1, marker] - lower_y) * inverse_y
        if periodic_y == 1:
            rel_y = periodic_position(rel_y, num_y)
        j0 = left_node(rel_y, num_y, periodic_y)
        j1 = right_node(j0, num_y, periodic_y)
        fy = cell_fraction(rel_y, j0)
        rel_z = (positions[2, marker] - lower_z) * inverse_z
        if periodic_z == 1:
            rel_z = periodic_position(rel_z, num_z)
        k0 = left_node(rel_z, num_z, periodic_z)
        k1 = right_node(k0, num_z, periodic_z)
        fz = cell_fraction(rel_z, k0)
        result[0, marker] = (
            edge_weight(fy, 0) * edge_weight(fz, 0) * edges[i0, j0, k0, 0]
            + edge_weight(fy, 1) * edge_weight(fz, 0) * edges[i0, j1, k0, 0]
            + edge_weight(fy, 0) * edge_weight(fz, 1) * edges[i0, j0, k1, 0]
            + edge_weight(fy, 1) * edge_weight(fz, 1) * edges[i0, j1, k1, 0]
        ) * inverse_x
        result[1, marker] = (
            edge_weight(fx, 0) * edge_weight(fz, 0) * edges[i0, j0, k0, 1]
            + edge_weight(fx, 1) * edge_weight(fz, 0) * edges[i1, j0, k0, 1]
            + edge_weight(fx, 0) * edge_weight(fz, 1) * edges[i0, j0, k1, 1]
            + edge_weight(fx, 1) * edge_weight(fz, 1) * edges[i1, j0, k1, 1]
        ) * inverse_y
        result[2, marker] = (
            edge_weight(fx, 0) * edge_weight(fy, 0) * edges[i0, j0, k0, 2]
            + edge_weight(fx, 1) * edge_weight(fy, 0) * edges[i1, j0, k0, 2]
            + edge_weight(fx, 0) * edge_weight(fy, 1) * edges[i0, j1, k0, 2]
            + edge_weight(fx, 1) * edge_weight(fy, 1) * edges[i1, j1, k0, 2]
        ) * inverse_z
