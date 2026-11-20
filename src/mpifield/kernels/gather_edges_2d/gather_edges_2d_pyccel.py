"""The edge (Whitney) field of a 2D grid at the markers.

The Pyccel version (CPU) of the ``gather_edges_2d`` kernel; ``gather_edges_2d_cuda.cu``
next to this file is the CUDA version (GPU) and ``gather_edges_2d_numpy.py`` the NumPy
version, with the same arguments. Loaded by `mpifield.kernels.catalog`; this file is
compiled with Pyccel there.

``edges`` ``(nx, ny, 2)`` holds the line integrals along the x-edges (``[..., 0]``)
and the y-edges (``[..., 1]``). In the cell of a marker (`LINEAR`'s cell: clamped on a
bounded axis, wrapped on a periodic one) ``E_x`` is linear in y between the cell's
lower and upper x-edge, ``E_y`` linear in x between its left and right y-edge, each
divided by the cell size along the edge. ``positions`` and ``result`` are ``(2, N)``.
"""

from mpifield.kernels.linear_cell_pyccel import (
    cell_fraction,
    left_node,
    periodic_position,
    right_node,
)
from mpifield.kernels.whitney_cell_pyccel import edge_weight


def gather_edges_2d(
    positions: "float[:,:]",
    edges: "float[:,:,:]",
    result: "float[:,:]",
    lower_x: float,
    lower_y: float,
    cell_size_x: float,
    cell_size_y: float,
    num_x: int,
    num_y: int,
    periodic_x: int,
    periodic_y: int,
) -> None:
    """Write the field of ``edges`` at ``positions`` into ``result``."""
    num_markers = positions.shape[1]
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    # fmt: off
    #$ omp parallel for private(rel_x, rel_y, i_left, i_right, j_left, j_right, fx, fy) schedule(dynamic, 4096)
    # fmt: on
    for marker in range(num_markers):
        rel_x = (positions[0, marker] - lower_x) * inverse_x
        if periodic_x == 1:
            rel_x = periodic_position(rel_x, num_x)
        i_left = left_node(rel_x, num_x, periodic_x)
        i_right = right_node(i_left, num_x, periodic_x)
        fx = cell_fraction(rel_x, i_left)
        rel_y = (positions[1, marker] - lower_y) * inverse_y
        if periodic_y == 1:
            rel_y = periodic_position(rel_y, num_y)
        j_left = left_node(rel_y, num_y, periodic_y)
        j_right = right_node(j_left, num_y, periodic_y)
        fy = cell_fraction(rel_y, j_left)
        result[0, marker] = (
            edge_weight(fy, 0) * edges[i_left, j_left, 0]
            + edge_weight(fy, 1) * edges[i_left, j_right, 0]
        ) * inverse_x
        result[1, marker] = (
            edge_weight(fx, 0) * edges[i_left, j_left, 1]
            + edge_weight(fx, 1) * edges[i_right, j_left, 1]
        ) * inverse_y
