"""The edge (Whitney) field of a 1D grid at the markers.

The Pyccel version (CPU) of the ``gather_edges_1d`` kernel; ``gather_edges_1d_cuda.cu``
next to this file is the CUDA version (GPU) and ``gather_edges_1d_numpy.py`` the NumPy
version, with the same arguments. Loaded by `mpifield.kernels.catalog`; this file is
compiled with Pyccel there.

``edges`` ``(nx, 1)`` holds the line integral along every cell; the field in the cell
of a marker (`LINEAR`'s cell: clamped on a bounded axis, wrapped on a periodic one) is
constant, the line integral divided by the cell size. ``positions`` and ``result``
are ``(1, N)``.
"""

from mpifield.kernels.linear_cell_pyccel import left_node, periodic_position


def gather_edges_1d(
    positions: "float[:,:]",
    edges: "float[:,:]",
    result: "float[:,:]",
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> None:
    """Write the field of ``edges`` at ``positions`` into ``result``."""
    num_markers = positions.shape[1]
    inverse_x = 1.0 / cell_size_x
    # fmt: off
    #$ omp parallel for private(rel_x, i_left) schedule(dynamic, 4096)
    # fmt: on
    for marker in range(num_markers):
        rel_x = (positions[0, marker] - lower_x) * inverse_x
        if periodic_x == 1:
            rel_x = periodic_position(rel_x, num_x)
        i_left = left_node(rel_x, num_x, periodic_x)
        result[0, marker] = edges[i_left, 0] * inverse_x
