"""The one-component trilinear deposit on contiguous rows, serial.

The Pyccel version (CPU) of the ``scatter_rows_3d`` kernel, the reference of the C version
``scatter_rows_3d_c.c`` (which runs on OpenMP threads, each into its own grid). Same nodes,
weights and operations as the one-component loop of ``scatter_linear_3d``: a marker with a
zero value deposits nothing, whatever its position.
"""

from mpifield.kernels.linear_cell_pyccel import (
    cell_fraction,
    clamped_left_node,
    next_node,
    wrapped_position,
)


def scatter_rows_3d(
    px: "float[:]",
    py: "float[:]",
    pz: "float[:]",
    values: "float[:]",
    field: "float[:]",
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
    """Add ``values`` at the positions ``(px, py, pz)`` to the nodes of ``field``.

    ``field`` is the grid ``(nx, ny, nz)`` flattened: node ``(i, j, k)`` is entry
    ``(i * ny + j) * nz + k``. The grid arguments are those of ``scatter_linear_3d``.
    """
    inverse_x = 1.0 / cell_size_x
    inverse_y = 1.0 / cell_size_y
    inverse_z = 1.0 / cell_size_z
    last_x = num_x - 2 + periodic_x
    last_y = num_y - 2 + periodic_y
    last_z = num_z - 2 + periodic_z
    for marker in range(px.shape[0]):
        single = values[marker]
        if single == 0.0:
            continue
        rel_x = wrapped_position((px[marker] - lower_x) * inverse_x, num_x, periodic_x)
        i_left = clamped_left_node(rel_x, last_x)
        fx = cell_fraction(rel_x, i_left)
        i_right = next_node(i_left, num_x)
        rel_y = wrapped_position((py[marker] - lower_y) * inverse_y, num_y, periodic_y)
        j_left = clamped_left_node(rel_y, last_y)
        fy = cell_fraction(rel_y, j_left)
        j_right = next_node(j_left, num_y)
        rel_z = wrapped_position((pz[marker] - lower_z) * inverse_z, num_z, periodic_z)
        k_left = clamped_left_node(rel_z, last_z)
        fz = cell_fraction(rel_z, k_left)
        k_right = next_node(k_left, num_z)
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
        field[r00 + k_left] += w00 * gz * single
        field[r00 + k_right] += w00 * fz * single
        field[r01 + k_left] += w01 * gz * single
        field[r01 + k_right] += w01 * fz * single
        field[r10 + k_left] += w10 * gz * single
        field[r10 + k_right] += w10 * fz * single
        field[r11 + k_left] += w11 * gz * single
        field[r11 + k_right] += w11 * fz * single
