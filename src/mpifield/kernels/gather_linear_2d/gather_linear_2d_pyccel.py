"""Bilinear interpolation of a 2D grid field at the markers.

On a periodic axis the ``num`` nodes cover ``num`` cells and the last node
neighbours the first; on a bounded axis positions are clamped to the first
and last cell.  The fraction (the weight of the right node) is clamped to
[0, 1].  The same rules as ``mpifield/kernels/linear_cell.cuh``.

The Pyccel version (CPU) of the ``gather_linear_2d`` kernel; ``gather_linear_2d_cuda.cu`` next to
this file is the CUDA version (GPU), with the same arguments.  Both are loaded
by `mpifield.kernels.catalog`; this file is compiled with Pyccel there.
"""

from mpifield.kernels.linear_cell_pyccel import (
    cell_fraction,
    left_node,
    periodic_position,
    right_node,
)


def gather_linear_2d(
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
) -> None:
    """Interpolate a multi-component 2-D grid field to marker positions.

    ``positions`` is ``(2, N)`` and ``result`` ``(components, N)`` (component-major).
    """
    num_markers = positions.shape[1]
    num_components = field.shape[2]
    # Multiplying by the inverse spacing, and clamping with comparisons: Pyccel
    # turns min/max into variadic C functions, a call each that the compiler
    # cannot inline.
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    # fmt: off
    #$ omp parallel for private(rel_x, rel_y, i_left, i_right, j_left, j_right, fx, fy, w_ll, w_lr, w_ul, w_ur, component) schedule(dynamic, 4096)
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
        w_ll = (1.0 - fx) * (1.0 - fy)
        w_lr = fx * (1.0 - fy)
        w_ul = (1.0 - fx) * fy
        w_ur = fx * fy
        for component in range(num_components):
            result[component, marker] = (
                w_ll * field[i_left, j_left, component]
                + w_lr * field[i_right, j_left, component]
                + w_ul * field[i_left, j_right, component]
                + w_ur * field[i_right, j_right, component]
            )
