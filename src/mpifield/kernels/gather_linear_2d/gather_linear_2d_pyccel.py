"""Bilinear interpolation of a 2D grid field at the markers.

On a periodic axis the ``num`` nodes cover ``num`` cells and the last node
neighbours the first; on a bounded axis positions are clamped to the first
and last cell.  The fraction (the weight of the right node) is clamped to
[0, 1].  The same rules as ``mpifield/kernels/linear_cell.cuh``.

The Pyccel version (CPU) of the ``gather_linear_2d`` kernel; ``gather_linear_2d_cuda.cu`` next to
this file is the CUDA version (GPU), with the same arguments.  Both are loaded
by :mod:`mpifield.kernels.catalog`; this file is compiled with Pyccel there.
"""


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
            if not (rel_x >= 0.0 and rel_x < num_x):
                rel_x = rel_x - num_x * (rel_x // num_x)
            i_left = int(rel_x)
            if i_left >= num_x:
                i_left = num_x - 1
            i_right = i_left + 1
            if i_right == num_x:
                i_right = 0
        else:
            i_left = int(rel_x) if rel_x >= 0.0 else 0
            if i_left > num_x - 2:
                i_left = num_x - 2
            i_right = i_left + 1
        fx = rel_x - i_left
        if fx < 0.0:
            fx = 0.0
        if fx > 1.0:
            fx = 1.0
        rel_y = (positions[1, marker] - lower_y) * inverse_y
        if periodic_y == 1:
            if not (rel_y >= 0.0 and rel_y < num_y):
                rel_y = rel_y - num_y * (rel_y // num_y)
            j_left = int(rel_y)
            if j_left >= num_y:
                j_left = num_y - 1
            j_right = j_left + 1
            if j_right == num_y:
                j_right = 0
        else:
            j_left = int(rel_y) if rel_y >= 0.0 else 0
            if j_left > num_y - 2:
                j_left = num_y - 2
            j_right = j_left + 1
        fy = rel_y - j_left
        if fy < 0.0:
            fy = 0.0
        if fy > 1.0:
            fy = 1.0
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
