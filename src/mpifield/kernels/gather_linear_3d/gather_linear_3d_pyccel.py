"""Trilinear interpolation of a 3D grid field at the markers.

The Pyccel version (CPU) of the ``gather_linear_3d`` kernel; ``gather_linear_3d_cuda.cu`` next to
this file is the CUDA version (GPU), with the same arguments.  Both are loaded
by `mpifield.kernels.catalog`; this file is compiled with Pyccel there.
"""

from mpifield.kernels.linear_cell_pyccel import (
    cell_fraction,
    left_node,
    periodic_position,
    right_node,
)


def gather_linear_3d(
    positions: "float[:,:]",
    field: "float[:,:]",
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
    """Interpolate a 3D grid field at the markers into ``result`` (components, N).

    ``positions`` is ``(3, N)`` (component-major). ``field`` is the grid
    ``(nx, ny, nz, components)`` reshaped to ``(nx * ny * nz, components)``: cell
    ``(i, j, k)`` is row ``(i * ny + j) * nz + k``.
    """
    num_markers = positions.shape[1]
    num_components = field.shape[1]
    # Multiplying by the inverse spacing, and clamping with comparisons: Pyccel
    # turns min/max into variadic C functions, a call each that the compiler
    # cannot inline.
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    inverse_z = 1.0 / cell_size_z
    # fmt: off
    #$ omp parallel for private(rel_x, i_left, i_right, fx, rel_y, j_left, j_right, fy, rel_z, k_left, k_right, fz, component, gx, gy, gz, w00, w01, w10, w11, r00, r01, r10, r11, s00, s01, s10, s11) schedule(dynamic, 2048)
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
        rel_z = (positions[2, marker] - lower_z) * inverse_z
        if periodic_z == 1:
            rel_z = periodic_position(rel_z, num_z)
        k_left = left_node(rel_z, num_z, periodic_z)
        k_right = right_node(k_left, num_z, periodic_z)
        fz = cell_fraction(rel_z, k_left)
        # the four (x, y) columns of the cell and their weights, once for every
        # component; each column holds the z nodes k_left, k_right
        gx = 1.0 - fx
        gy = 1.0 - fy
        gz = 1.0 - fz
        w00 = gx * gy
        w01 = gx * fy
        w10 = fx * gy
        w11 = fx * fy
        r00 = (i_left * num_y + j_left) * num_z
        r01 = (i_left * num_y + j_right) * num_z
        r10 = (i_right * num_y + j_left) * num_z
        r11 = (i_right * num_y + j_right) * num_z
        for component in range(num_components):
            s00 = (
                gz * field[r00 + k_left, component]
                + fz * field[r00 + k_right, component]
            )
            s01 = (
                gz * field[r01 + k_left, component]
                + fz * field[r01 + k_right, component]
            )
            s10 = (
                gz * field[r10 + k_left, component]
                + fz * field[r10 + k_right, component]
            )
            s11 = (
                gz * field[r11 + k_left, component]
                + fz * field[r11 + k_right, component]
            )
            result[component, marker] = w00 * s00 + w01 * s01 + w10 * s10 + w11 * s11
