"""The spline edge weightings of degree 1 to 3: `QUADRATIC_EDGES`, `CUBIC_EDGES`."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from mpifield import (
    CUBIC,
    CUBIC_EDGES,
    LINEAR,
    QUADRATIC,
    QUADRATIC_EDGES,
    WHITNEY,
    BlockGrid,
)
from mpifield.kernels.catalog import KERNELS
from mpifield.kernels.spline_edges import (
    along,
    deposit_paths_spline,
    gather_edges_spline,
)

NODES = {1: LINEAR, 2: QUADRATIC, 3: CUBIC}
EDGES = {1: WHITNEY, 2: QUADRATIC_EDGES, 3: CUBIC_EDGES}
CASES = [(ndim, periodic) for ndim in (1, 2, 3) for periodic in (False, True)]


def _case(ndim, periodic, seed=0, n=200, reach=3.0):
    """A grid, paths partly outside it (some of several cells), charges, an edge field."""
    rng = np.random.default_rng(seed)
    shape = (9, 8, 7)[:ndim]
    lower = np.array([0.0, -1.0, 2.0])[:ndim]
    spacing = np.array([0.25, 0.5, 0.1])[:ndim]
    grid = BlockGrid.from_geometry(lower, spacing, shape, (periodic,) * ndim)
    upper = lower + spacing * (np.array(shape) - 1)
    start = rng.uniform(lower - 3 * spacing, upper + 3 * spacing, (n, ndim)).T.copy()
    end = start + rng.normal(scale=reach * spacing[:, None], size=start.shape)
    charges = rng.normal(size=n)
    edges = rng.normal(size=(*shape, ndim))
    return grid, start, end, charges, edges


@pytest.mark.parametrize(("ndim", "periodic"), CASES)
def test_degree_one_is_whitney(ndim, periodic):
    grid, start, end, charges, edges = _case(ndim, periodic)
    expected = np.zeros_like(edges)
    WHITNEY.deposit_paths_block(start, end, charges, expected, grid)
    found = np.zeros_like(edges)
    for implementation in ("pyccel", "numpy"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            found[...] = 0.0
            KERNELS[f"deposit_paths_spline_{ndim}d"](
                start, end, charges, found, *grid.geometry, 1
            )
        np.testing.assert_allclose(found, expected, rtol=0, atol=1e-13)
    gathered = np.zeros_like(start)
    WHITNEY.gather_block(start, edges, gathered, grid)
    result = np.zeros_like(start)
    KERNELS[f"gather_edges_spline_{ndim}d"](start, edges, result, *grid.geometry, 1)
    np.testing.assert_allclose(result, gathered, rtol=0, atol=1e-12)


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize(("ndim", "periodic"), CASES)
def test_the_path_deposit_conserves_the_charge_of_the_node_weighting(
    degree, ndim, periodic
):
    """G^T j = rho(ends) - rho(starts), with the folds and clamps of bounded axes."""
    grid, start, end, charges, _ = _case(ndim, periodic, seed=degree)
    shape = grid.num_gridpoints
    current = np.zeros((*shape, ndim))
    EDGES[degree].deposit_paths_block(start, end, charges, current, grid)
    before = np.zeros((*shape, 1))
    NODES[degree].scatter_block(start, charges, before, grid)
    after = np.zeros((*shape, 1))
    NODES[degree].scatter_block(end, charges, after, grid)
    divergence = EDGES[degree].divergence_block(current, grid)
    np.testing.assert_allclose(
        np.asarray(divergence).reshape(shape),
        (after - before)[..., 0],
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize(("ndim", "periodic"), [(1, False), (2, True), (3, False)])
def test_the_current_is_the_transpose_of_the_gather(degree, ndim, periodic):
    """j . e = sum q * integral of E . dl along the clamped paths (a fine midpoint rule)."""
    grid, start, end, charges, edges = _case(ndim, periodic, seed=5, n=12)
    current = np.zeros_like(edges)
    EDGES[degree].deposit_paths_block(start, end, charges, current, grid)
    geometry = grid.geometry
    lower = np.asarray(geometry[:ndim], dtype=float)[:, None]
    spacing = np.asarray(geometry[ndim : 2 * ndim], dtype=float)[:, None]
    nodes = np.asarray(geometry[2 * ndim : 3 * ndim], dtype=float)[:, None]
    upper = lower + spacing * (nodes - 1)

    def clamp(points):
        return points if periodic else np.clip(points, lower, upper)

    samples = 4000
    t = np.linspace(0.0, 1.0, samples + 1)
    work = 0.0
    for path in range(charges.size):
        line = clamp(start[:, path : path + 1] + t * (end - start)[:, path : path + 1])
        middle = 0.5 * (line[:, 1:] + line[:, :-1])
        field = np.zeros_like(middle)
        EDGES[degree].gather_block(middle, edges, field, grid)
        work += charges[path] * np.sum(field * np.diff(line, axis=1))
    assert float(np.sum(current * edges)) == pytest.approx(work, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(("ndim", "periodic"), CASES)
def test_compiled_edge_kernels_match_numpy(degree, ndim, periodic):
    grid, start, end, charges, edges = _case(ndim, periodic, seed=7, reach=8.0)
    charges[::11] = 0.0
    end[:, ::13] = np.nan  # deposits nothing
    results = {}
    for implementation in ("pyccel", "numpy"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            current = np.zeros_like(edges)
            KERNELS[f"deposit_paths_spline_{ndim}d"](
                start, end, charges, current, *grid.geometry, degree
            )
            field = np.zeros_like(start)
            KERNELS[f"gather_edges_spline_{ndim}d"](
                start, edges, field, *grid.geometry, degree
            )
        results[implementation] = (current, field)
    for compiled, reference in zip(results["pyccel"], results["numpy"], strict=True):
        np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-12)


def test_a_uniform_field_is_gathered_exactly():
    """An edge field of equal line integrals is a uniform field, with every degree.

    Away from the bounded walls: within the stencil's reach of one, an edge past the wall
    does not exist (what the fold of the node charge leaves of it), so a uniform field is
    gathered smaller there, as the fold of the node weights breaks the first moment.
    """
    grid = BlockGrid.from_geometry([0.0, 0.0], [0.5, 0.25], [8, 9], [False, True])
    edges = np.zeros((8, 9, 2))
    edges[:-1, :, 0] = 0.5 * 3.0  # E_x = 3 on every x edge (none after the last node)
    edges[:, :, 1] = 0.25 * -2.0
    points = np.random.default_rng(3).uniform([1.0, -1.0], [2.5, 3.0], (50, 2)).T
    for weighting in (WHITNEY, QUADRATIC_EDGES, CUBIC_EDGES):
        field = np.zeros_like(points)
        weighting.gather_block(points.copy(), edges, field, grid)
        np.testing.assert_allclose(field[0], 3.0, atol=1e-12)
        np.testing.assert_allclose(field[1], -2.0, atol=1e-12)


def test_edges_past_a_bounded_wall_do_not_exist():
    edges, weights, exists = along(np.array([0.0, 7.0]), 8, False, 3)
    assert [list(e) for e in exists] == [[False, True], [True, True], [True, False]]
    np.testing.assert_allclose(sum(weights), 1.0)
    with pytest.raises(ValueError, match="spline edge degrees"):
        gather_edges_spline(
            np.zeros((1, 1)),
            np.zeros((4, 1)),
            np.zeros((1, 1)),
            [0.0],
            [1.0],
            [4],
            [False],
            0,
        )
    with pytest.raises(ValueError, match="spline edge degrees"):
        deposit_paths_spline(
            np.zeros((1, 1)),
            np.ones((1, 1)),
            np.ones(1),
            np.zeros((4, 1)),
            [0.0],
            [1.0],
            [4],
            [False],
            4,
        )


def test_the_edge_weightings_know_their_degree():
    assert (WHITNEY.degree, QUADRATIC_EDGES.degree, CUBIC_EDGES.degree) == (1, 2, 3)
    assert (
        repr(QUADRATIC_EDGES)
        == "EdgeWeighting('quadratic', ndim=(1, 2, 3), halo=2, degree=2)"
    )
