"""The SymPy rules and convergence problems of mpifield's method cards (for the tests).

The rules of the shape functions run the compiled gather and scatter kernels of
`LINEAR` and `QUADRATIC` (through `Weighting.gather_block` and
`Weighting.scatter_block`), not the NumPy cell functions that carry the cards: a
gather of the unit field on node ``k`` at a marker returns that node's weight
``W_k``. SymPy is imported only inside the rules, so the package does not need it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from methodcard import ConvergenceProblem, Rule

#: Nodes of the 1D test grids of the shape-function rules (spacing 1, lower bound 0).
NODES = 11

#: The node whose weight the shape rules read: far enough from both edges for a
#: quadratic stencil around any position within 3/2 of it.
CENTRE = 5


def _weighting(name: str) -> Any:
    """The built-in weighting called ``name`` (``"linear"`` or ``"quadratic"``)."""
    from mpifield.weighting import LINEAR, QUADRATIC

    return {"linear": LINEAR, "quadratic": QUADRATIC}[name]


def _grid(num_nodes: tuple[int, ...], periodic: bool) -> Any:
    """A block grid of spacing 1 from 0 on."""
    from mpifield._types import BlockGrid

    ndim = len(num_nodes)
    return BlockGrid.from_geometry(
        (0.0,) * ndim, (1.0,) * ndim, num_nodes, (periodic,) * ndim
    )


def gather_1d(
    name: str, rel: Any, node_values: Any, periodic: bool = False
) -> np.ndarray:
    """Gather ``node_values`` (one per node, spacing 1) at the relative positions ``rel``."""
    rel = np.atleast_1d(np.asarray(rel, dtype=float))
    values = np.asarray(node_values, dtype=float)
    result = np.zeros((1, rel.size))
    _weighting(name).gather_block(
        rel[None, :], values[:, None], result, _grid((values.size,), periodic)
    )
    return result[0]


def _unit(node: int, num_nodes: int = NODES) -> np.ndarray:
    """The field that is 1 on ``node`` and 0 elsewhere."""
    values = np.zeros(num_nodes)
    values[node] = 1.0
    return values


def _shape(name: str) -> Callable[[Any], Any]:
    """The one-dimensional shape function ``w(xi)`` of a weighting, in SymPy."""
    import sympy

    half, three_halves = sympy.Rational(1, 2), sympy.Rational(3, 2)

    def linear(xi: Any) -> Any:
        return sympy.Piecewise((1 - sympy.Abs(xi), sympy.Abs(xi) <= 1), (0, True))

    def quadratic(xi: Any) -> Any:
        return sympy.Piecewise(
            (sympy.Rational(3, 4) - xi**2, sympy.Abs(xi) <= half),
            (half * (three_halves - sympy.Abs(xi)) ** 2, sympy.Abs(xi) <= three_halves),
            (0, True),
        )

    return {"linear": linear, "quadratic": quadratic}[name]


def shape_rules(name: str) -> dict[str, Rule]:
    """The rules of a shape function, checked with the compiled gather kernels.

    - ``shape``: the weight of a node is ``w(r - k)``;
    - ``partition``: the weights of all nodes sum to ``sum_k w(r - k)`` (= 1);
    - ``first_moment``: ``sum_k W_k k = r``, so linear fields are gathered exactly;
    - ``bounded``: the weight of node 0 near and beyond a bounded edge (the fold);
    - ``periodic``: the weight of node 0 for markers on both sides of the seam.
    """
    import sympy

    w = _shape(name)
    xi, r = sympy.symbols("xi r", real=True)
    reach = 1.0 if name == "linear" else 1.5
    interior = (reach, NODES - 1 - reach)
    if name == "linear":
        edge = sympy.Piecewise((1, r < 0), (1 - r, True))
        edge_range = (-1.0, 1.0)
    else:
        edge = sympy.Piecewise(
            (sympy.Rational(7, 8), r < 0),
            (
                sympy.Rational(1, 2) * (sympy.Rational(1, 2) - r) ** 2
                + sympy.Rational(3, 4)
                - r**2,
                True,
            ),
        )
        edge_range = (-1.0, 0.5)

    def node_weight(xi: float) -> float:
        return float(gather_1d(name, CENTRE + xi, _unit(CENTRE))[0])

    def total_weight(r: float) -> float:
        return float(gather_1d(name, r, np.ones(NODES))[0])

    def mean_node(r: float) -> float:
        return float(gather_1d(name, r, np.arange(NODES, dtype=float))[0])

    def edge_weight(r: float) -> float:
        return float(gather_1d(name, r, _unit(0))[0])

    def seam_weight(xi: float) -> float:
        return float(gather_1d(name, xi, _unit(0, 8), periodic=True)[0])

    return {
        "shape": Rule(w(xi), node_weight, {"xi": (-reach - 0.25, reach + 0.25)}),
        "partition": Rule(
            sum(w(r - k) for k in range(NODES)), total_weight, {"r": interior}
        ),
        "first_moment": Rule(r, mean_node, {"r": interior}, rtol=1e-12),
        "bounded": Rule(edge, edge_weight, {"r": edge_range}),
        "periodic": Rule(w(xi), seam_weight, {"xi": (-reach, reach)}),
    }


def gather_problem(name: str) -> ConvergenceProblem:
    """The gather of a smooth periodic function: the error at random points.

    ``f(x) = sin(2 pi x) + cos(4 pi x) / 2`` on ``[0, 1)``, sampled on ``n`` nodes of a
    periodic axis and gathered at 2000 fixed random points, for ``h = 1 / n`` from
    1/32 to 1/256; the error is the largest one.
    """
    points = np.random.default_rng(0).uniform(0.0, 1.0, 2000)

    def f(x: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)

    def error(spacing: float) -> float:
        n = round(1.0 / spacing)
        nodes = f(np.arange(n) / n)
        gathered = gather_1d(name, points * n, nodes, periodic=True)
        return float(np.max(np.abs(gathered - f(points))))

    return ConvergenceProblem(error, steps=[1 / 32, 1 / 64, 1 / 128, 1 / 256])


def scatter_rules() -> dict[str, Rule]:
    """The rules of the deposit, with the compiled scatter kernels.

    - ``linear_2d``: one `LINEAR` marker at ``(i + f_x, j + f_y)`` puts
      ``q (1 - f_x)(1 - f_y)`` on node ``(i, j)``;
    - ``total``: three `QUADRATIC` markers, two of them beyond the bounded edges,
      deposit their whole value.
    """
    import sympy

    fx, fy, q = sympy.symbols("f_x f_y q", real=True)
    q1, q2, q3 = sympy.symbols("q_1 q_2 q_3", real=True)

    def linear_2d(f_x: float, f_y: float, q: float) -> float:
        field = np.zeros((6, 7, 1))
        position = np.array([[2.0 + f_x], [3.0 + f_y]])
        _weighting("linear").scatter_block(
            position, np.array([[q]]), field, _grid((6, 7), False)
        )
        return float(field[2, 3, 0])

    def total(q_1: float, q_2: float, q_3: float) -> float:
        field = np.zeros((NODES, 1))
        positions = np.array([[-0.4, 4.3, NODES - 0.8]])
        _weighting("quadratic").scatter_block(
            positions, np.array([[q_1, q_2, q_3]]), field, _grid((NODES,), False)
        )
        return float(field.sum())

    return {
        "linear_2d": Rule(
            q * (1 - fx) * (1 - fy),
            linear_2d,
            {"f_x": (0.0, 1.0), "f_y": (0.0, 1.0), "q": (-2.0, 2.0)},
        ),
        "total": Rule(q1 + q2 + q3, total, {"q_1": (0.5, 2.0)}, rtol=1e-12),
    }


def gather_rules() -> dict[str, Rule]:
    """The rule of the gather, with the compiled 3D `QUADRATIC` kernel.

    ``quadratic_3d``: a marker at ``(4, 4, 4) + delta`` gathers the unit field on node
    ``(4, 4, 4)`` as ``prod_d (3/4 - delta_d^2)``.
    """
    import sympy

    dx, dy, dz = sympy.symbols("delta_x delta_y delta_z", real=True)
    rule = (
        (sympy.Rational(3, 4) - dx**2)
        * (sympy.Rational(3, 4) - dy**2)
        * (sympy.Rational(3, 4) - dz**2)
    )

    def quadratic_3d(delta_x: float, delta_y: float, delta_z: float) -> float:
        field = np.zeros((9, 9, 9, 1))
        field[4, 4, 4, 0] = 1.0
        position = 4.0 + np.array([[delta_x], [delta_y], [delta_z]])
        result = np.zeros((1, 1))
        _weighting("quadratic").gather_block(
            position, field, result, _grid((9, 9, 9), False)
        )
        return float(result[0, 0])

    half = (-0.49, 0.49)
    return {
        "quadratic_3d": Rule(
            rule,
            quadratic_3d,
            {"delta_x": half, "delta_y": half, "delta_z": half},
        )
    }


def gradient_problem() -> ConvergenceProblem:
    """`Field.gradient` of ``sin(2 pi x)`` on a periodic axis, the largest error.

    The central difference is ``sin(2 pi h) / h cos(2 pi x)``, so the error is
    ``(2 pi)^3 h^2 / 6`` to leading order.
    """
    from mpifield.field import Field

    def error(spacing: float) -> float:
        n = round(1.0 / spacing)
        field = Field(
            ndim=1,
            lower_bound=(0.0,),
            upper_bound=(1.0,),
            num_gridpoints=(n,),
            periodic=True,
        )
        x = np.asarray(field.get_grid_points()[0])
        field.set_field_data(np.sin(2 * np.pi * x))
        derivative = np.asarray(field.gradient().to_numpy())[:, 0]
        return float(np.max(np.abs(derivative - 2 * np.pi * np.cos(2 * np.pi * x))))

    return ConvergenceProblem(error, steps=[1 / 32, 1 / 64, 1 / 128, 1 / 256])


def electric_field_problem() -> ConvergenceProblem:
    """`electric_field_from_potential` of ``sin(3 x)`` on ``[0, 1]``, inside ``[1/4, 3/4]``.

    The interior error of the central difference, away from the first-order ends.
    """
    from mpifield.prescribed import electric_field_from_potential

    def error(spacing: float) -> float:
        n = round(1.0 / spacing) + 1
        field = electric_field_from_potential(
            lambda x: np.sin(3.0 * x), (0.0,), (1.0,), n
        )
        x = np.linspace(0.0, 1.0, n)
        inside = (x >= 0.25 - 1e-12) & (x <= 0.75 + 1e-12)
        e = np.asarray(field.to_numpy())[:, 0]
        return float(np.max(np.abs(e[inside] + 3.0 * np.cos(3.0 * x[inside]))))

    return ConvergenceProblem(error, steps=[1 / 16, 1 / 32, 1 / 64, 1 / 128])


def axial_potential_rules() -> dict[str, Rule]:
    """`axial_potential` of the table ``(0, phi_0), (1, phi_1)`` along ``x``.

    ``table``: linear between the points, the end values outside.
    """
    import sympy

    x, phi0, phi1 = sympy.symbols("x phi_0 phi_1", real=True)
    rule = sympy.Piecewise(
        (phi0, x < 0), (phi0 + (phi1 - phi0) * x, x <= 1), (phi1, True)
    )

    def table(x: float, phi_0: float, phi_1: float) -> float:
        from mpifield.prescribed import axial_potential

        potential = axial_potential([0.0, 1.0], [phi_0, phi_1])
        return float(np.asarray(potential(np.array([x])))[0])

    ranges = {"x": (-0.5, 1.5), "phi_0": (-5.0, 5.0), "phi_1": (-5.0, 5.0)}
    return {"table": Rule(rule, table, ranges)}


def _field_1d(values: Any, upper: float, periodic: bool) -> Any:
    """A one-component 1D `Field` on ``[0, upper]`` holding ``values``."""
    from mpifield.field import Field

    values = np.asarray(values, dtype=float)
    field = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(upper,),
        num_gridpoints=(values.size,),
        periodic=periodic,
    )
    field.set_field_data(values)
    return field


def smoothing_rules() -> dict[str, Rule]:
    """The rules of `Field.smooth`, one binomial pass on five nodes.

    - ``filter``: node 2 becomes ``(g_1 + 2 g_2 + g_3) / 4``;
    - ``boundary``: on a bounded axis node 0 becomes ``(2 g_0 + g_1) / 4`` (zero
      outside); on a periodic one node 0 also takes node 4.
    """
    import sympy

    g0, g1, g2, g3, g4 = sympy.symbols("g_0 g_1 g_2 g_3 g_4", real=True)

    def inner(g_1: float, g_2: float, g_3: float) -> float:
        field = _field_1d([0.3, g_1, g_2, g_3, -0.7], 1.0, periodic=False)
        return float(np.asarray(field.smooth("binomial").to_numpy())[2, 0])

    def edge(g_0: float, g_1: float, g_4: float) -> float:
        bounded = _field_1d([g_0, g_1, 0.5, 0.2, g_4], 1.0, periodic=False)
        periodic = _field_1d([g_0, g_1, 0.5, 0.2, g_4], 1.0, periodic=True)
        smooth = [
            float(np.asarray(f.smooth("binomial").to_numpy())[0, 0])
            for f in (bounded, periodic)
        ]
        # both edge values in one number: the periodic one weighted by pi
        return smooth[0] + np.pi * smooth[1]

    edge_rule = (2 * g0 + g1) / 4 + sympy.Float(np.pi) * (g4 + 2 * g0 + g1) / 4
    return {
        "filter": Rule((g1 + 2 * g2 + g3) / 4, inner, rtol=1e-13),
        "boundary": Rule(edge_rule, edge, rtol=1e-13),
    }


def gradient_rules() -> dict[str, Rule]:
    """The rules of `Field.gradient` on five nodes of spacing ``h`` (a bounded axis).

    - ``central``: node 2 gets ``(g_3 - g_1) / (2 h)``;
    - ``ends``: node 0 gets ``(g_1 - g_0) / h``.
    """
    import sympy

    g0, g1, g3, h = sympy.symbols("g_0 g_1 g_3 h", real=True)

    def node(index: int, values: list[float], h: float) -> float:
        field = _field_1d(values, 4 * h, periodic=False)
        return float(np.asarray(field.gradient().to_numpy())[index, 0])

    def central(g_1: float, g_3: float, h: float) -> float:
        return node(2, [0.4, g_1, 1.3, g_3, 0.2], h)

    def ends(g_0: float, g_1: float, h: float) -> float:
        return node(0, [g_0, g_1, 1.3, 0.1, 0.2], h)

    spacing = {"h": (0.1, 2.0)}
    return {
        "central": Rule((g3 - g1) / (2 * h), central, spacing),
        "ends": Rule((g1 - g0) / h, ends, spacing),
    }


def evaluation_rules() -> dict[str, Rule]:
    """The rule of `evaluate_field` across a periodic seam.

    ``seam``: four nodes on ``[0, 1)``; between the last node (``x = 3/4``) and the
    upper bound the field is linear from ``g_3`` to ``g_0``.
    """
    import sympy

    x, g0, g3 = sympy.symbols("x g_0 g_3", real=True)

    def seam(x: float, g_0: float, g_3: float) -> float:
        field = _field_1d([g_0, 1.0, 2.0, g_3], 1.0, periodic=True)
        return float(field.evaluate(x=x))

    rule = g3 + (g0 - g3) * (x - sympy.Rational(3, 4)) * 4
    return {"seam": Rule(rule, seam, {"x": (0.75, 1.0)})}
