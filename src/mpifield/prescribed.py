"""Electric fields prescribed through an electrostatic potential."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cunumpy as xp
from maybempi import MPI
from methodcard import method

from mpifield import _method_checks
from mpifield._types import Array
from mpifield.field import Field

Potential = Callable[..., Array]


@method(
    equations={
        r"table": r"\phi(\mathbf x) = \phi_j + (\phi_{j+1} - \phi_j) \frac{x_a - z_j}{z_{j+1} - z_j} \ \ (z_j \le x_a \le z_{j+1}), \qquad \phi_0 \text{ below } z_0, \quad \phi_{n-1} \text{ above } z_{n-1}",
    },
    assumptions=(
        "$x_a$ the coordinate along `axis`; at least two strictly increasing table positions $z_j$",
        "potentials in V, positions in m",
    ),
    symbolic=_method_checks.axial_potential_rules,
)
def axial_potential(
    positions: Sequence[float],
    values: Sequence[float],
    axis: int = 0,
) -> Potential:
    """Return the potential (V) that depends on ``axis`` through a table.

    ``values`` are the potentials at the increasing ``positions`` (m); between
    them the potential is interpolated linearly, outside them it is constant.
    The result takes the coordinate arrays of the grid, as
    `electric_field_from_potential` calls it.
    """
    table_positions = xp.asarray(positions, dtype=float)
    table_values = xp.asarray(values, dtype=float)
    if table_positions.ndim != 1 or table_positions.shape != table_values.shape:
        raise ValueError("positions and values must be 1D arrays of equal length")
    if table_positions.size < 2 or xp.any(xp.diff(table_positions) <= 0.0):
        raise ValueError("positions must be strictly increasing, at least two")

    def potential(*coordinates: Array) -> Array:
        return xp.interp(coordinates[axis], table_positions, table_values)

    return potential


@method(
    equations={
        r"field": r"\mathbf E = -\nabla \phi",
        r"differences": r"E_{d,k} = -\frac{\phi_{k+1} - \phi_{k-1}}{2 h_d} \text{ inside}, \qquad E_{d,0} = -\frac{\phi_1 - \phi_0}{h_d}, \quad E_{d,N-1} = -\frac{\phi_{N-1} - \phi_{N-2}}{h_d}",
    },
    order=2,
    assumptions=(
        r"$\phi$ evaluated on the nodes; second order inside, first order at both ends of every axis, also a periodic one (no wrapping, unlike `Field.gradient`)",
        "the order is that of the interior nodes",
        "no Poisson equation is solved: the field is prescribed",
    ),
    problem=_method_checks.electric_field_problem,
)
def electric_field_from_potential(
    potential: Potential,
    lower_bound: tuple[float, ...],
    upper_bound: tuple[float, ...],
    num_gridpoints: int | tuple[int, ...],
    comm: MPI.Comm | None = None,
    **field_options: object,
) -> Field:
    """Return ``E = -grad(phi)`` on a grid for a prescribed potential.

    ``potential(*coordinates)`` receives one array per axis, each with the
    shape of the grid (indexing ``"ij"``), and returns the potential in V.  The
    field has one component per axis, in V/m, from second-order central
    differences (one-sided at the grid ends).  No Poisson equation is solved:
    the field is fixed, e.g. for test particles.
    ``field_options`` go to `Field`.
    """
    ndim = len(lower_bound)
    grid = Field(
        ndim=ndim,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_gridpoints=num_gridpoints,
        comm=comm,
        **field_options,
    )
    coordinates = xp.meshgrid(*grid.get_grid_points(), indexing="ij")
    phi = xp.asarray(potential(*coordinates), dtype=float)
    if phi.shape != tuple(grid.num_gridpoints):
        raise ValueError(
            f"potential returned shape {phi.shape}, expected {grid.num_gridpoints}"
        )
    components = [
        -xp.gradient(phi, spacing, axis=axis)
        for axis, spacing in enumerate(grid.grid_spacing)
    ]
    return Field.from_array(
        xp.stack(components, axis=-1),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        comm=comm,
        **field_options,
    )
