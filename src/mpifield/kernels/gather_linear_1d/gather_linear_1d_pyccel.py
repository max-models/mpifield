"""Linear interpolation of a 1D grid field at the markers.

The Pyccel version (CPU) of the ``gather_linear_1d`` kernel; ``gather_linear_1d_cuda.cu`` next to
this file is the CUDA version (GPU), with the same arguments.  Both are loaded
by `mpifield.kernels.catalog`; this file is compiled with Pyccel there.
"""

from mpifield.kernels.linear_cell_pyccel import (
    cell_fraction,
    left_node,
    periodic_position,
    right_node,
)


def gather_linear_1d(
    positions: "float[:,:]",
    field: "float[:,:]",
    result: "float[:,:]",
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> None:
    """Interpolate ``field`` (nx, components) at the markers into ``result``.

    ``positions`` is ``(1, N)`` and ``result`` ``(components, N)`` (component-major).
    """
    num_markers = positions.shape[1]
    num_components = field.shape[1]
    # Multiplying by the inverse spacing, and clamping with comparisons: Pyccel
    # turns min/max into variadic C functions, a call each that the compiler
    # cannot inline.
    inverse_x = 1.0 / cell_size_x
    # fmt: off
    #$ omp parallel for private(rel_x, i_left, i_right, fx, component) schedule(dynamic, 4096)
    # fmt: on
    for marker in range(num_markers):
        rel_x = (positions[0, marker] - lower_x) * inverse_x
        if periodic_x == 1:
            rel_x = periodic_position(rel_x, num_x)
        i_left = left_node(rel_x, num_x, periodic_x)
        i_right = right_node(i_left, num_x, periodic_x)
        fx = cell_fraction(rel_x, i_left)
        for component in range(num_components):
            result[component, marker] = (1.0 - fx) * field[
                i_left, component
            ] + fx * field[i_right, component]
