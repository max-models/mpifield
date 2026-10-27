"""Weightings: the built-in ones, a user-defined one, and the kernel convention."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI

from mpifield import LINEAR, QUADRATIC, BlockGrid, Field, Weighting


def _field(ndim, periodic=None, components=1, **kwargs):
    return Field(
        ndim=ndim,
        lower_bound=(0.0,) * ndim,
        upper_bound=(1.0,) * ndim,
        num_gridpoints=(11, 9, 7)[:ndim],
        num_components=components,
        periodic=periodic,
        **kwargs,
    )


def _nearest_node_2d(positions, values, field, *geometry):
    """A user-defined deposit by the kernel convention: all on the nearest node."""
    lower, cell, num = geometry[0:2], geometry[2:4], geometry[4:6]
    nodes = [
        np.clip(np.rint((positions[d] - lower[d]) / cell[d]).astype(int), 0, num[d] - 1)
        for d in range(2)
    ]
    np.add.at(field, (nodes[0], nodes[1]), values.T)


def _nearest_node_gather_2d(positions, field, result, *geometry):
    """The gather of :func:`_nearest_node_2d`: the value of the nearest node."""
    lower, cell, num = geometry[0:2], geometry[2:4], geometry[4:6]
    nodes = [
        np.clip(np.rint((positions[d] - lower[d]) / cell[d]).astype(int), 0, num[d] - 1)
        for d in range(2)
    ]
    result[...] = field[nodes[0], nodes[1]].T


NEAREST = Weighting(
    "nearest",
    scatter={2: _nearest_node_2d},
    gather={2: _nearest_node_gather_2d},
    halo=1,
)


def test_a_user_defined_weighting_deposits_and_gathers():
    field = _field(2, components=2)
    positions = np.array([[0.31, 0.69], [0.5, 0.06]])  # (2, N): nodes (3, 4), (7, 0)
    field.scatter(positions, np.array([[1.0, 2.0], [3.0, 4.0]]), NEAREST)
    data = field.to_numpy()
    assert data[3, 4].tolist() == [1.0, 3.0]
    assert data[7, 0].tolist() == [2.0, 4.0]
    assert data.sum() == 10.0
    gathered = np.asarray(xp.to_numpy(field.gather(positions, NEAREST)))
    np.testing.assert_array_equal(gathered, [[1.0, 2.0], [3.0, 4.0]])


def test_a_weighting_without_the_dimension_raises():
    with pytest.raises(ValueError, match="nearest weighting has no 3D scatter kernel"):
        _field(3).scatter(np.full((3, 1), 0.5), 1.0, NEAREST)
    with pytest.raises(ValueError, match=r"no 1D gather kernel \(has 2D\)"):
        _field(1).gather(np.array([[0.5]]), NEAREST)


def test_weighting_checks_its_arguments():
    with pytest.raises(ValueError, match="halo must be >= 0"):
        Weighting("bad", scatter={}, gather={}, halo=-1)
    with pytest.raises(ValueError, match="1, 2 or 3 dimensions, got 4"):
        Weighting("bad", scatter={4: _nearest_node_2d}, gather={})
    assert repr(NEAREST) == "Weighting('nearest', ndim=(2,), halo=1)"
    assert (LINEAR.halo, QUADRATIC.halo) == (1, 2)


def test_the_kernel_mappings_are_read_only():
    with pytest.raises(TypeError):
        LINEAR.scatter[2] = _nearest_node_2d  # type: ignore[index]


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("weighting", [LINEAR, QUADRATIC], ids=lambda w: w.name)
def test_constant_and_linear_fields_are_gathered_exactly_inside(ndim, weighting):
    """Both weightings sum to one and reproduce a linear function away from the edges."""
    field = _field(ndim)
    nodes = np.meshgrid(
        *(np.linspace(0.0, 1.0, n) for n in field.num_gridpoints), indexing="ij"
    )
    field.set_field_data(2.0 + sum((d + 1) * x for d, x in enumerate(nodes)))
    rng = np.random.default_rng(3)
    points = rng.uniform(0.2, 0.8, (ndim, 300))
    gathered = np.asarray(xp.to_numpy(field.gather(points, weighting)))[0]
    expected = 2.0 + sum((d + 1) * points[d] for d in range(ndim))
    np.testing.assert_allclose(gathered, expected, rtol=1e-12)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_quadratic_deposit_keeps_all_the_charge(ndim, periodic):
    """Also for markers at and beyond the edges of a bounded axis."""
    field = _field(ndim, (periodic,) * ndim, components=2)
    rng = np.random.default_rng(4)
    points = rng.uniform(-0.1, 1.1, (ndim, 500))
    points[:, :50] = rng.choice([0.0, 1.0], (ndim, 50))
    values = rng.normal(size=(2, 500))
    field.scatter(points, values, QUADRATIC)
    np.testing.assert_allclose(
        field.local_with_halos.reshape(-1, 2).sum(axis=0), values.sum(axis=1)
    )


def test_quadratic_weights_of_a_marker_on_a_node_and_between_two():
    field = _field(1)
    dx = float(field.grid_spacing[0])
    field.scatter(np.array([[5 * dx]]), 1.0, QUADRATIC)  # on node 5
    field.scatter(np.array([[2.5 * dx]]), 8.0, QUADRATIC)  # midway: nodes 2 and 3
    np.testing.assert_allclose(
        field.to_numpy()[:, 0], [0, 0, 4, 4, 0.125, 0.75, 0.125, 0, 0, 0, 0]
    )


def test_quadratic_stencil_folds_onto_the_edge_node_of_a_bounded_axis():
    field = _field(1)
    field.scatter(np.array([[0.0]]), 1.0, QUADRATIC)  # node 0: 1/8 falls off the edge
    np.testing.assert_allclose(field.to_numpy()[:3, 0], [0.875, 0.125, 0.0])


def test_quadratic_wraps_on_a_periodic_axis():
    field = _field(1, periodic=(True,))
    field.scatter(np.array([[0.0]]), 1.0, QUADRATIC)
    data = field.to_numpy()[:, 0]
    np.testing.assert_allclose(data[[-1, 0, 1]], [0.125, 0.75, 0.125])


def test_1d_linear_deposit_past_a_bounded_end_stays_on_the_edge_node():
    """It used to wrap around: a marker past the upper end put its charge on node 0."""
    field = _field(1)
    field.scatter(np.array([[1.2, -0.3]]), np.array([1.0, 2.0]))
    data = field.to_numpy()[:, 0]
    assert data[-1] == 1.0
    assert data[0] == 2.0
    assert data.sum() == 3.0


def test_1d_linear_gather_wraps_on_a_periodic_axis():
    """It used to clamp: the last cell read the last node only."""
    field = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(1.0,),
        num_gridpoints=4,
        periodic=True,
    )
    field.set_field_data(np.array([0.0, 1.0, 2.0, 3.0]))
    gathered = np.asarray(xp.to_numpy(field.gather(np.array([[0.875, 1.125]]))))
    np.testing.assert_allclose(gathered, [[1.5, 0.5]])


def test_values_must_match_the_components():
    field = _field(2, components=3)
    with pytest.raises(ValueError, match="Expected 3 components, got 2"):
        field.scatter(np.full((2, 4), 0.5), np.ones((2, 4)))
    field.scatter(np.full((2, 4), 0.5), 1.0)  # a number goes to every component
    np.testing.assert_allclose(field.to_numpy().sum(axis=(0, 1)), [4.0, 4.0, 4.0])


def test_the_block_grid_inside_the_domain_leaves_out_halos_beyond_bounded_edges():
    field = _field(2, periodic=(True, False), comm=MPI.COMM_SELF, num_ghost_points=2)
    whole = field.block_grid(cpu=True)
    inside = field.block_grid(cpu=True, inside_domain=True)
    assert tuple(whole.num_gridpoints) == (15, 13)
    assert whole.index == (slice(0, 15), slice(0, 13))
    # the periodic axis keeps its halo nodes (the wrapped nodes), the bounded one not
    assert tuple(inside.num_gridpoints) == (15, 9)
    assert inside.index == (slice(0, 15), slice(2, 11))
    assert inside.geometry[1] == pytest.approx(0.0)  # lower y: the domain edge
    assert inside.geometry[0] == pytest.approx(-2 * float(field.grid_spacing[0]))
    assert field.local_with_halos[inside.index].shape == (15, 9, 1)


def test_block_grid_from_geometry_runs_a_weighting_on_any_array():
    grid = BlockGrid.from_geometry((0.0, 0.0), (0.5, 0.5), (3, 3), (True, False))
    assert grid.geometry == (0.0, 0.0, 0.5, 0.5, 3, 3, 1, 0)
    array = np.zeros((3, 3, 1))
    QUADRATIC.scatter_block(np.array([[0.0], [0.5]]), np.array([[1.0]]), array, grid)
    np.testing.assert_allclose(array[:, :, 0].sum(axis=1), [0.75, 0.125, 0.125])
    with pytest.raises(ValueError, match="differ in length"):
        BlockGrid.from_geometry((0.0,), (0.5, 0.5), (3, 3))
