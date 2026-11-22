"""The spline kernels of degree 0 to 3, `CUBIC` and `Splines`."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from mpifield import (
    CUBIC,
    LINEAR,
    QUADRATIC,
    SPLINES,
    WHITNEY,
    BlockGrid,
    EdgeWeighting,
    Splines,
    Weighting,
    splines,
)
from mpifield.kernels.catalog import KERNELS
from mpifield.kernels.spline_cell import spline_cell

CASES = [
    (1, (False,)),
    (1, (True,)),
    (2, (False, True)),
    (2, (True, True)),
    (3, (False, False, False)),
    (3, (True, False, True)),
]


def _case(ndim, periodic, seed=0, n=400):
    """A block grid, markers partly outside it, two components of values and a field."""
    rng = np.random.default_rng(seed)
    shape = (7, 6, 5)[:ndim]
    lower = np.array([0.0, -1.0, 2.0])[:ndim]
    spacing = np.array([0.25, 0.5, 0.1])[:ndim]
    upper = lower + spacing * np.array(shape)
    grid = BlockGrid.from_geometry(lower, spacing, shape, periodic)
    points = rng.uniform(lower - 0.3, upper + 0.3, (n, ndim)).T.copy()
    values = rng.normal(size=(2, n))
    field = rng.normal(size=(*shape, 2))
    return grid, points, values, field


def _scatter(degree, ndim, points, values, shape, grid):
    out = np.zeros((*shape, values.shape[0]))
    KERNELS[f"scatter_spline_{ndim}d"](points, values, out, *grid.geometry, degree)
    return out


def _gather(degree, ndim, points, field, grid):
    out = np.zeros((field.shape[-1], points.shape[1]))
    KERNELS[f"gather_spline_{ndim}d"](points, field, out, *grid.geometry, degree)
    return out


@pytest.mark.parametrize("implementation", ["pyccel", "numpy"])
@pytest.mark.parametrize(("ndim", "periodic"), CASES)
@pytest.mark.parametrize(("weighting", "degree"), [(LINEAR, 1), (QUADRATIC, 2)])
def test_spline_kernels_reproduce_the_linear_and_quadratic_ones(
    implementation, ndim, periodic, weighting, degree
):
    grid, points, values, field = _case(ndim, periodic)
    expected = np.zeros_like(field)
    weighting.scatter_block(points, values, expected, grid)
    gathered = np.zeros_like(values)
    weighting.gather_block(points, field, gathered, grid)
    with xp.kernels.use_host_kernel_implementation(implementation):
        deposit = _scatter(degree, ndim, points, values, field.shape[:-1], grid)
        gather = _gather(degree, ndim, points, field, grid)
    np.testing.assert_allclose(deposit, expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(gather, gathered, rtol=0, atol=1e-13)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize(("ndim", "periodic"), CASES)
def test_compiled_spline_kernels_match_numpy_and_are_adjoint(degree, ndim, periodic):
    grid, points, values, field = _case(ndim, periodic, seed=degree)
    values[:, ::9] = 0.0
    points[:, ::9] = np.nan  # dead markers may sit anywhere
    shape = field.shape[:-1]
    deposit = _scatter(degree, ndim, points, values, shape, grid)
    with xp.kernels.use_host_kernel_implementation("numpy"):
        reference = _scatter(degree, ndim, points, values, shape, grid)
    np.testing.assert_allclose(deposit, reference, rtol=0, atol=1e-13)
    # no charge is lost, also from markers beyond the bounded edges
    np.testing.assert_allclose(deposit.reshape(-1, 2).sum(axis=0), values.sum(axis=1))

    alive = ~np.isnan(points[0])
    alive_points = points[:, alive].copy()  # standard strides for a direct call
    gathered = _gather(degree, ndim, alive_points, field, grid)
    with xp.kernels.use_host_kernel_implementation("numpy"):
        expected = _gather(degree, ndim, alive_points, field, grid)
    np.testing.assert_allclose(gathered, expected, rtol=0, atol=1e-13)
    # sum_i g(x_i) q_i = sum_n G_n rho_n, component by component
    np.testing.assert_allclose(
        (gathered * values[:, alive]).sum(axis=1),
        (field * deposit).reshape(-1, 2).sum(axis=0),
        rtol=1e-12,
    )


def test_degree_zero_is_the_nearest_grid_point():
    grid = BlockGrid.from_geometry([0.0], [1.0], [6], [False])
    points = np.array([[0.2, 1.49, 1.51, 4.7, 9.0, -3.0]])
    deposit = _scatter(0, 1, points, np.ones((1, 6)), (6,), grid)
    np.testing.assert_array_equal(deposit[:, 0], [2, 1, 1, 0, 0, 2])


@pytest.mark.parametrize("periodic", [False, True])
def test_cubic_weights_are_the_cubic_b_spline(periodic):
    """Interior weights of a marker at fraction d of a cell, and their two moments."""
    d = np.linspace(0.0, 0.99, 12)
    nodes, weights = spline_cell(3.0 + d, 10, periodic, 3)
    assert [int(n[0]) for n in nodes] == [2, 3, 4, 5]
    np.testing.assert_allclose(weights[0], (1 - d) ** 3 / 6)
    np.testing.assert_allclose(weights[3], d**3 / 6)
    np.testing.assert_allclose(sum(weights), 1.0)
    np.testing.assert_allclose(
        sum(w * n for w, n in zip(weights, nodes, strict=True)), 3.0 + d
    )
    # the second moment of the cubic B-spline is 1/3
    second = sum(w * (n - (3.0 + d)) ** 2 for w, n in zip(weights, nodes, strict=True))
    np.testing.assert_allclose(second, 1.0 / 3.0)


def test_cubic_stencil_folds_at_a_bounded_edge_and_wraps_at_a_seam():
    nodes, weights = spline_cell(np.array([0.0, -2.0, 0.5]), 8, False, 3)
    np.testing.assert_array_equal(np.array(nodes)[:, 0], [0, 0, 1, 2])
    np.testing.assert_allclose(np.array(weights)[:, 1], [1 / 6, 2 / 3, 1 / 6, 0.0])
    nodes, _ = spline_cell(np.array([7.5, -0.25]), 8, True, 3)
    np.testing.assert_array_equal(np.array(nodes), [[6, 6], [7, 7], [0, 0], [1, 1]])


def test_spline_cell_rejects_other_degrees():
    with pytest.raises(ValueError, match="spline degrees"):
        spline_cell(np.zeros(1), 4, False, 4)


def test_splines_pair_the_weightings_of_each_degree():
    assert tuple(SPLINES) == (1, 2, 3)
    assert splines(1).nodes is LINEAR and splines(1).edges is WHITNEY
    assert splines(2).nodes is QUADRATIC and splines(2).edges is None
    assert splines(3).nodes is CUBIC
    assert splines(splines(2)) is splines(2)
    assert [splines(p).halo for p in (1, 2, 3)] == [1, 2, 2]
    assert splines(1).require_edges() is WHITNEY
    assert repr(splines(1)) == "Splines(degree=1, nodes='linear', edges='whitney')"


def test_splines_check_degrees():
    with pytest.raises(ValueError, match="no built-in splines of degree 5"):
        splines(5)
    with pytest.raises(ValueError, match="an integer"):
        splines(1.5)
    with pytest.raises(ValueError, match="an integer"):
        splines(True)
    with pytest.raises(ValueError, match="degree 2 has no edge weighting yet"):
        splines(2).require_edges()
    with pytest.raises(ValueError, match="has degree 1, not 2"):
        Splines(2, LINEAR)
    with pytest.raises(ValueError, match="has degree 1, not 2"):
        Splines(2, QUADRATIC, WHITNEY)
    with pytest.raises(ValueError, match="has degree None, not 1"):
        Splines(1, Weighting("mine", scatter={}, gather={}))
    with pytest.raises(ValueError, match="degree must be >= 1"):
        EdgeWeighting("bad", gather={}, deposit_paths={}, degree=0)
