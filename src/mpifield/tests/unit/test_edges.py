"""The edge field of `WHITNEY`: gather, path deposit, gradient and divergence on a block."""

from __future__ import annotations

import itertools

import cunumpy as xp
import numpy as np
import pytest

from mpifield import LINEAR, WHITNEY, BlockGrid, EdgeWeighting
from mpifield.kernels.whitney_cell import _pieces

SHAPES = {1: (7,), 2: (6, 5), 3: (5, 4, 4)}
CASES = [
    (ndim, periodic)
    for ndim in (1, 2, 3)
    for periodic in itertools.product((False, True), repeat=ndim)
]


def _grid(ndim, periodic, seed=0):
    rng = np.random.default_rng(seed)
    lower = rng.uniform(-1.0, 1.0, ndim)
    spacing = rng.uniform(0.2, 0.7, ndim)
    return BlockGrid.from_geometry(lower, spacing, SHAPES[ndim], periodic)


def _paths(grid, count, seed=1, length=1.0):
    """Random paths starting around the grid (some outside), of about ``length``."""
    rng = np.random.default_rng(seed)
    lower = grid.bounds.lower_bound
    upper = grid.bounds.upper_bound
    ndim = lower.size
    start = rng.uniform(lower - 0.5, upper + 0.5, (count, ndim)).T.copy()
    end = start + rng.normal(0.0, length, (ndim, count))
    charges = rng.normal(size=count)
    return start, end, charges


def _shape(grid):
    return tuple(int(n) for n in grid.num_gridpoints)


def _linear_charge(points, charges, grid):
    """`LINEAR`'s deposit of the charges at ``points`` on the nodes, ``num_nodes``."""
    charge = np.zeros((*_shape(grid), 1))
    LINEAR.scatter_block(points, charges, charge, grid)
    return charge[..., 0]


def _deposit(grid, start, end, charges):
    edges = np.zeros((*_shape(grid), len(grid.periodic)))
    WHITNEY.deposit_paths_block(start, end, charges, edges, grid)
    return edges


@pytest.mark.parametrize(("ndim", "periodic"), CASES)
def test_the_deposit_conserves_linear_charge(ndim, periodic):
    grid = _grid(ndim, periodic)
    start, end, charges = _paths(grid, 400)
    current = _deposit(grid, start, end, charges)
    change = _linear_charge(end, charges, grid) - _linear_charge(start, charges, grid)
    np.testing.assert_allclose(
        WHITNEY.divergence_block(current, grid), change, atol=1e-12
    )


@pytest.mark.parametrize(("ndim", "periodic"), CASES)
def test_bounded_axes_leave_the_last_edges_empty(ndim, periodic):
    grid = _grid(ndim, periodic)
    current = _deposit(grid, *_paths(grid, 200, length=3.0))
    for axis, wraps in enumerate(periodic):
        last = current.take(-1, axis=axis)[..., axis]
        if wraps:
            assert np.abs(last).max() > 0.0
        else:
            assert np.all(last == 0.0)


@pytest.mark.parametrize(("ndim", "periodic"), CASES)
def test_numpy_and_pyccel_kernels_agree(ndim, periodic):
    grid = _grid(ndim, periodic, seed=4)
    start, end, charges = _paths(grid, 300, seed=5)
    charges[::7] = 0.0
    start[:, ::7] = np.nan  # a dead marker: zero charge, any position
    end[0, 3] = np.inf
    rng = np.random.default_rng(6)
    edges = rng.normal(size=(*_shape(grid), ndim))
    points = _paths(grid, 300, seed=7)[0]
    results = {}
    for implementation in ("pyccel", "numpy"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            current = _deposit(grid, start, end, charges)
            field = np.zeros((ndim, points.shape[1]))
            WHITNEY.gather_block(points, edges, field, grid)
            results[implementation] = current, field
    for pyccel, numpy in zip(results["pyccel"], results["numpy"], strict=True):
        np.testing.assert_allclose(pyccel, numpy, rtol=1e-12, atol=1e-12)


def test_dead_and_non_finite_paths_deposit_nothing():
    grid = _grid(2, (False, True))
    start = np.array([[0.1, np.nan, 0.2], [0.3, 0.1, np.inf]])
    end = start + 0.4
    current = _deposit(grid, start, end, np.array([0.0, 1.0, 2.0]))
    assert np.all(current == 0.0)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_many_paths_over_threads_match_one_thread(ndim, monkeypatch):
    """Enough paths for the per-thread copies of the compiled kernels."""
    grid = _grid(ndim, (True,) * ndim)
    start, end, charges = _paths(grid, 25000, seed=8)
    current = _deposit(grid, start, end, charges)
    with xp.kernels.use_host_kernel_implementation("numpy"):
        reference = _deposit(grid, start, end, charges)
    np.testing.assert_allclose(current, reference, atol=1e-10)


@pytest.mark.parametrize(("ndim", "periodic"), CASES)
def test_the_field_of_a_gradient_is_the_gradient_of_linear(ndim, periodic):
    grid = _grid(ndim, periodic, seed=9)
    rng = np.random.default_rng(10)
    potential = rng.normal(size=_shape(grid))
    edges = WHITNEY.gradient_block(potential, grid)
    # points inside cells, away from their faces: the multilinear interpolant is
    # linear along each axis there, so a central difference is exact
    cells = np.array([rng.integers(0, n - 1, 50) for n in _shape(grid)])
    fraction = rng.uniform(0.2, 0.8, cells.shape)
    spacing = grid.grid_spacing[:, None]
    points = grid.bounds.lower_bound[:, None] + (cells + fraction) * spacing
    field = np.zeros_like(points)
    WHITNEY.gather_block(points, edges, field, grid)
    step = 0.05
    for axis in range(ndim):
        shift = np.zeros((ndim, 1))
        shift[axis] = step * spacing[axis]
        values = []
        for sign in (1.0, -1.0):
            value = np.zeros((1, points.shape[1]))
            LINEAR.gather_block(
                points + sign * shift, potential[..., None], value, grid
            )
            values.append(value[0])
        derivative = (values[0] - values[1]) / (2.0 * shift[axis])
        np.testing.assert_allclose(field[axis], derivative, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("ndim", [2, 3])
def test_the_current_times_the_field_is_the_work(ndim):
    """``j . e`` is the line integral of the field along the paths, times the charges."""
    grid = _grid(ndim, (True,) * ndim, seed=11)
    start, end, charges = _paths(grid, 5, seed=12, length=0.8)
    edges = np.random.default_rng(13).normal(size=(*_shape(grid), ndim))
    current = _deposit(grid, start, end, charges)
    # the field along the pieces of the paths in their cells (at most quadratic
    # along a piece): three Gauss points per piece integrate it exactly
    lower = grid.bounds.lower_bound[:, None]
    spacing = grid.grid_spacing[:, None]
    path, a, b = _pieces((start - lower) / spacing, (end - lower) / spacing)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    work = 0.0
    for node, weight in zip(nodes, weights, strict=True):
        points = lower + spacing * (a + 0.5 * (node + 1.0) * (b - a))
        field = np.zeros_like(points)
        WHITNEY.gather_block(points, edges, field, grid)
        step = (b - a) * spacing
        work += 0.5 * weight * np.sum(charges[path] * np.sum(field * step, axis=0))
    assert np.sum(current * edges) == pytest.approx(work, rel=1e-12)


def test_a_path_outside_a_bounded_axis_moves_along_the_wall():
    grid = BlockGrid.from_geometry((0.0, 0.0), (1.0, 1.0), (4, 4), (False, False))
    # from (0.5, -1) to (2.5, -1): clamped onto the lower wall y = 0
    current = _deposit(
        grid, np.array([[0.5], [-1.0]]), np.array([[2.5], [-1.0]]), [1.0]
    )
    np.testing.assert_allclose(current[:3, 0, 0], [0.5, 1.0, 0.5])
    assert np.count_nonzero(current) == 3


def test_gradient_and_divergence_are_transposes():
    grid = _grid(3, (True, False, True))
    rng = np.random.default_rng(14)
    potential = rng.normal(size=_shape(grid))
    current = rng.normal(size=(*_shape(grid), 3))
    for axis in range(3):
        if not grid.periodic[axis]:
            index = [slice(None)] * 4
            index[axis] = -1
            index[3] = axis
            current[tuple(index)] = 0.0
    left = np.sum(WHITNEY.gradient_block(potential, grid) * current)
    right = np.sum(potential * WHITNEY.divergence_block(current, grid))
    assert left == pytest.approx(right, rel=1e-12)


def test_shape_and_argument_errors():
    grid = _grid(2, (False, False))
    with pytest.raises(ValueError, match="edge field on this grid"):
        WHITNEY.gather_block(
            np.zeros((2, 1)), np.zeros((6, 5, 1)), np.zeros((2, 1)), grid
        )
    with pytest.raises(ValueError, match="paths from"):
        WHITNEY.deposit_paths_block(
            np.zeros((2, 3)), np.zeros((2, 2)), np.ones(3), np.zeros((6, 5, 2)), grid
        )
    with pytest.raises(ValueError, match="halo"):
        EdgeWeighting("bad", gather={}, deposit_paths={}, halo=-1)
    with pytest.raises(ValueError, match="1, 2 or 3"):
        EdgeWeighting("bad", gather={4: print}, deposit_paths={})
    partial = EdgeWeighting("partial", gather={2: print}, deposit_paths={})
    with pytest.raises(ValueError, match="no 3D gather kernel .has 2D"):
        partial.gather_kernel(3)
    with pytest.raises(ValueError, match="has none"):
        partial.deposit_paths_kernel(2)
    assert repr(WHITNEY) == "EdgeWeighting('whitney', ndim=(1, 2, 3), halo=1)"


def test_the_numpy_edge_weights_are_the_pyccel_ones():
    from mpifield.kernels import whitney_cell, whitney_cell_pyccel

    rng = np.random.default_rng(30)
    for _ in range(20):
        f, g, du, dw = rng.uniform(-1.0, 1.0, 4)
        for upper in (0, 1):
            assert whitney_cell.edge_weight(
                f, upper
            ) == whitney_cell_pyccel.edge_weight(f, upper)
            for other in (0, 1):
                assert whitney_cell.edge_path_weight(
                    f, g, du, dw, upper, other
                ) == pytest.approx(
                    whitney_cell_pyccel.edge_path_weight(f, g, du, dw, upper, other),
                    rel=1e-15,
                )
