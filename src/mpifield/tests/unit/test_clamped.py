"""Clamped splines: the clamped kernels, `splines(p, clamped=True)` and `mpifield.bsplines`.

The reference is `scipy.interpolate.BSpline` throughout, also for the folded and periodic
weights (the SciPy part of the cross-check of the splines plan).
"""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest
from scipy.interpolate import BSpline

from mpifield import (
    CLAMPED_SPLINES,
    LINEAR,
    WHITNEY,
    BlockGrid,
    bsplines,
    splines,
)
from mpifield.kernels.catalog import KERNELS
from mpifield.kernels.spline_cell import clamped_cell, spline_cell


def _knots(num, degree):
    """The clamped knot vector in cells."""
    return np.concatenate(
        [np.zeros(degree), np.arange(num, dtype=float), np.full(degree, num - 1.0)]
    )


def _matrix(columns, weights, rows, size):
    out = np.zeros((rows, size))
    for column, weight in zip(columns, weights, strict=True):
        np.add.at(out, (np.arange(rows), column), weight)
    return out


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_the_clamped_stencils_are_scipy_s_b_splines(degree):
    num = 7
    knots = _knots(num, degree)
    x = np.linspace(0.0, num - 1.0, 400)[1:-1] + 0.0013  # off the knots
    x = x[x < num - 1]
    size = num + degree - 1
    nodes = _matrix(*clamped_cell(x, num, degree, False), x.size, size)
    np.testing.assert_allclose(
        nodes, BSpline.design_matrix(x, knots, degree).toarray(), atol=1e-14
    )
    staggered = _matrix(*clamped_cell(x, num, degree, True), x.size, size - 1)
    for i in range(1, size):
        element = BSpline.basis_element(knots[i : i + degree + 1], extrapolate=False)
        scale = degree / (knots[i + degree] - knots[i])
        np.testing.assert_allclose(
            staggered[:, i - 1], scale * np.nan_to_num(element(x)), atol=1e-14
        )
    # B_i' = D_(i-1) - D_i: the derivative identity behind charge conservation
    derivative = np.stack(
        [BSpline(knots, np.eye(size)[i], degree).derivative()(x) for i in range(size)],
        axis=1,
    )
    padded = np.pad(staggered, ((0, 0), (1, 1)))
    np.testing.assert_allclose(derivative, padded[:, :-1] - padded[:, 1:], atol=1e-12)


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("periodic", [0, 1])
def test_the_folded_and_periodic_weights_are_uniform_b_splines_inside(degree, periodic):
    """Away from the walls the weights are SciPy's uniform B-spline of the degree."""
    num = 12
    x = np.random.default_rng(degree).uniform(4.0, 7.0, 200)
    nodes, weights = spline_cell(x, num, periodic, degree)
    element = BSpline.basis_element(np.arange(degree + 2) - (degree + 1) / 2)
    for node, weight in zip(nodes, weights, strict=True):
        np.testing.assert_allclose(weight, element(x - node), atol=1e-14)


def _clamped_case(ndim, degree, seed=0):
    rng = np.random.default_rng(seed)
    num = np.array((8, 7, 6)[:ndim])
    spacing = np.array((0.5, 0.4, 0.3)[:ndim])
    clamped = [True, False, True][:ndim]
    periodic = [False, True, False][:ndim]
    grid = BlockGrid.from_geometry(np.zeros(ndim), spacing, num, periodic, clamped)
    size = tuple(
        int(n) + (degree - 1 if c else 0) for n, c in zip(num, clamped, strict=True)
    )
    upper = spacing * (num - 1)
    points = np.ascontiguousarray(rng.uniform(-0.2, upper + 0.2, (300, ndim)).T)
    return rng, grid, size, periodic, points


def _previous(values, axis, periodic):
    """The values at ``k - 1`` along ``axis``; zero before the first on a bounded axis."""
    if periodic:
        return np.roll(values, 1, axis=axis)
    return np.pad(
        values, [(1, 0) if a == axis else (0, 0) for a in range(values.ndim)]
    )[tuple(slice(0, -1) if a == axis else slice(None) for a in range(values.ndim))]


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_clamped_axes_keep_the_charge_and_conserve_it_along_paths(ndim, degree):
    rng, grid, size, periodic, points = _clamped_case(ndim, degree)
    shape = splines(degree, clamped=True)
    charges = rng.normal(size=points.shape[1])
    ends = np.ascontiguousarray(
        points + rng.normal(scale=2 * grid.grid_spacing[:, None], size=points.shape)
    )
    results = {}
    for implementation in ("pyccel", "numpy"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            before = np.zeros((*size, 1))
            shape.nodes.scatter_block(points, charges, before, grid)
            after = np.zeros((*size, 1))
            shape.nodes.scatter_block(ends, charges, after, grid)
            current = np.zeros((*size, ndim))
            shape.edges.deposit_paths_block(points, ends, charges, current, grid)
        results[implementation] = (before, after, current)
    for compiled, reference in zip(results["pyccel"], results["numpy"], strict=True):
        np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-12)
    before, after, current = results["pyccel"]
    # the clamped splines are a partition of unity up to the walls: no charge is lost
    assert before.sum() == pytest.approx(charges.sum(), abs=1e-12)
    divergence = sum(
        _previous(current[..., a], a, periodic[a]) - current[..., a]
        for a in range(ndim)
    )
    np.testing.assert_allclose(divergence, (after - before)[..., 0], atol=1e-12)


def test_a_clamped_gather_is_the_tensor_product_of_scipy_s_b_splines():
    degree, num, spacing = 3, np.array([8, 7]), np.array([0.5, 0.4])
    grid = BlockGrid.from_geometry([0.0, 0.0], spacing, num, clamped=[True, True])
    rng = np.random.default_rng(4)
    coefficients = rng.normal(size=(num[0] + degree - 1, num[1] + degree - 1))
    points = np.ascontiguousarray(
        rng.uniform(0.0, 0.999, (200, 2)).T * ((num - 1) * spacing)[:, None]
    )
    found = np.zeros((1, 200))
    splines(degree, clamped=True).nodes.gather_block(
        points, coefficients[..., None], found, grid
    )
    bx = BSpline.design_matrix(points[0], _knots(num[0], degree) * spacing[0], degree)
    by = BSpline.design_matrix(points[1], _knots(num[1], degree) * spacing[1], degree)
    expected = np.einsum("ki,ij,kj->k", bx.toarray(), coefficients, by.toarray())
    np.testing.assert_allclose(found[0], expected, atol=1e-13)


@pytest.mark.parametrize("degree", [2, 3])
def test_clamped_faces_of_the_curl_gather_the_curl_of_the_edge_field(degree):
    """The commuting property on clamped axes, up to the walls (central differences)."""
    rng = np.random.default_rng(degree)
    num, spacing = np.array([7, 6, 6]), np.array([0.5, 0.4, 0.3])
    grid = BlockGrid.from_geometry(np.zeros(3), spacing, num, clamped=[True] * 3)
    shape = splines(degree, clamped=True)
    size = tuple(int(n) + degree - 1 for n in num)
    edges = rng.normal(size=(*size, 3))
    for axis in range(3):  # a clamped axis has one staggered function fewer
        index = [slice(None)] * 3
        index[axis] = -1
        edges[(*index, axis)] = 0.0

    def following(values, axis):
        return np.pad(values, [(0, 1) if a == axis else (0, 0) for a in range(3)])[
            tuple(slice(1, None) if a == axis else slice(None) for a in range(3))
        ]

    ex, ey, ez = edges[..., 0], edges[..., 1], edges[..., 2]
    faces = np.stack(
        [
            ey + following(ez, 1) - following(ey, 2) - ez,
            ez + following(ex, 2) - following(ez, 0) - ex,
            ex + following(ey, 0) - following(ex, 1) - ey,
        ],
        axis=-1,
    )
    area = np.array(
        [spacing[1] * spacing[2], spacing[0] * spacing[2], spacing[0] * spacing[1]]
    )
    upper = spacing * (num - 1)
    points = np.ascontiguousarray(
        rng.uniform(1e-3, 1.0, (60, 3)).T * (upper[:, None] - 2e-3)
    )
    field = np.zeros((3, 60))
    shape.faces.gather_block(points, faces, field, grid)

    def edge_field(x):
        result = np.zeros((3, x.shape[1]))
        shape.edges.gather_block(x, edges, result, grid)
        return result

    step = 1e-7
    d = []
    for axis in range(3):
        shift = np.zeros((3, 1))
        shift[axis] = step
        d.append((edge_field(points + shift) - edge_field(points - shift)) / (2 * step))
    curl = np.array([d[1][2] - d[2][1], d[2][0] - d[0][2], d[0][1] - d[1][0]])
    np.testing.assert_allclose(field / area[:, None], curl, atol=1e-5)


def test_the_clamped_form_kernels_match_numpy():
    rng, grid, size, _, points = _clamped_case(3, 3, seed=9)
    for form, components in ((1, 3), (2, 3), (3, 1)):
        values = rng.normal(size=(components, points.shape[1]))
        field = rng.normal(size=(*size, components))
        found = {}
        for implementation in ("pyccel", "numpy"):
            with xp.kernels.use_host_kernel_implementation(implementation):
                deposit = np.zeros_like(field)
                KERNELS["scatter_forms_spline_3d"](
                    points, values, deposit, *grid.geometry, 3, form
                )
                gathered = np.zeros_like(values)
                KERNELS["gather_forms_spline_3d"](
                    points, field, gathered, *grid.geometry, 3, form
                )
            found[implementation] = (deposit, gathered)
        for compiled, reference in zip(found["pyccel"], found["numpy"], strict=True):
            np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-12)


def test_clamped_grids_and_the_weightings_that_take_them():
    grid = BlockGrid.from_geometry(
        [0.0, 0.0], [1.0, 1.0], [5, 4], [False, True], [True, False]
    )
    assert grid.geometry[-2:] == (2, 1)
    assert grid.clamped == (True, False)
    assert BlockGrid.from_geometry([0.0], [1.0], [5]).clamped == (False,)
    with pytest.raises(ValueError, match="either periodic or clamped"):
        BlockGrid.from_geometry([0.0], [1.0], [5], [True], [True])
    with pytest.raises(ValueError, match="differ in length"):
        BlockGrid.from_geometry([0.0], [1.0], [5], None, [True, False])
    points, values = np.zeros((2, 1)), np.ones((1, 1))
    with pytest.raises(ValueError, match="linear weighting does not take clamped axes"):
        LINEAR.scatter_block(points, values, np.zeros((5, 4, 1)), grid)
    with pytest.raises(
        ValueError, match="whitney weighting does not take clamped axes"
    ):
        WHITNEY.gather_block(points, np.zeros((5, 4, 2)), np.zeros((2, 1)), grid)
    with pytest.raises(ValueError, match="an edge field on this grid is"):
        splines(2, clamped=True).edges.gather_block(
            points, np.zeros((5, 4, 2)), np.zeros((2, 1)), grid
        )
    assert tuple(CLAMPED_SPLINES) == (1, 2, 3)
    assert all(s.clamped for s in CLAMPED_SPLINES.values())
    assert repr(splines(1, clamped=True)).endswith("clamped=True)")
    assert splines(splines(2, clamped=True), clamped=True) is splines(2, clamped=True)
    with pytest.raises(ValueError, match="does not take clamped axes"):
        splines(splines(1), clamped=True)
    with pytest.raises(ValueError, match="a clamped axis needs a degree"):
        spline_cell(np.zeros(1), 4, 2, 0)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_the_1d_module_is_scipy_s_clamped_spline_space(degree):
    lower, spacing, num = 0.3, 0.25, 9
    knots = bsplines.clamped_knots(lower, spacing, num, degree)
    size = bsplines.num_functions(num, degree)
    assert size == num + degree - 1 == len(knots) - degree - 1
    assert bsplines.num_functions(num, degree, 1) == size - 1
    x = np.linspace(lower, lower + spacing * (num - 1), 401)[:-1] + 1e-9
    np.testing.assert_allclose(
        bsplines.evaluate(x, lower, spacing, num, degree).toarray(),
        BSpline.design_matrix(x, knots, degree).toarray(),
        atol=1e-14,
    )
    # the derivative of a form-0 field is the form-1 field of G phi, exactly
    phi = np.random.default_rng(degree).normal(size=size)
    gradient = bsplines.evaluate(x, lower, spacing, num, degree, 1) @ (
        bsplines.incidence(num, degree) @ phi
    )
    np.testing.assert_allclose(
        gradient, BSpline(knots, phi, degree).derivative()(x), atol=1e-11
    )
    # the stiffness is G^T M_1 G, and the mass matrices hold the integrals
    g = bsplines.incidence(num, degree)
    stiffness = g.T @ bsplines.mass_matrix(lower, spacing, num, degree, 1) @ g
    points, weights = bsplines.quadrature(lower, spacing, num, degree + 2)
    derivatives = np.stack(
        [
            BSpline(knots, np.eye(size)[i], degree).derivative()(points)
            for i in range(size)
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        stiffness.toarray(),
        derivatives.T @ (weights[:, None] * derivatives),
        atol=1e-11,
    )
    mass = bsplines.mass_matrix(lower, spacing, num, degree)
    assert mass.sum() == pytest.approx(spacing * (num - 1))  # partition of unity


def test_the_1d_module_checks_its_arguments():
    with pytest.raises(ValueError, match="degrees are 1, 2 and 3"):
        bsplines.num_functions(5, 4)
    with pytest.raises(ValueError, match="forms are 0 and 1"):
        bsplines.num_functions(5, 2, 2)
    with pytest.raises(ValueError, match="at least 2 nodes"):
        bsplines.clamped_knots(0.0, 1.0, 1, 2)
    with pytest.raises(ValueError, match="at least one point"):
        bsplines.quadrature(0.0, 1.0, 5, 0)
