"""Markers and fields must agree on the node layout of periodic axes.

A periodic axis with ``n`` nodes has ``n`` cells over its length (the last
node neighbours the first).  The solvers always used that layout; the grid,
the charge deposit and the field gather used ``n - 1`` cells, which put the
nodes in the wrong places and left the two seam rows with half the charge.
"""

from __future__ import annotations

import numpy as np
import pytest
from simgeometry.grid import Grid

from mpifield import Field


def test_grid_spacing_uses_n_cells_on_a_periodic_axis() -> None:
    grid = Grid(
        ndim=2,
        lower_bound=(0.0, 0.0),
        upper_bound=(1.0, 2.0),
        num_gridpoints=(11, 8),
        periodic=(False, True),
    )
    assert grid.grid_spacing[0] == pytest.approx(1.0 / 10)
    assert grid.grid_spacing[1] == pytest.approx(2.0 / 8)
    # Without the flag nothing changes.
    plain = Grid(
        ndim=2, lower_bound=(0.0, 0.0), upper_bound=(1.0, 2.0), num_gridpoints=(11, 8)
    )
    assert plain.grid_spacing[1] == pytest.approx(2.0 / 7)


def _field(periodic: tuple[bool, ...], shape: tuple[int, ...]) -> Field:
    ndim = len(shape)
    return Field(
        ndim=ndim,
        num_components=1,
        lower_bound=(0.0,) * ndim,
        upper_bound=(1.0,) * ndim,
        num_gridpoints=shape,
        periodic=periodic,
    )


def test_deposit_at_cell_centres_is_uniform_across_the_periodic_seam() -> None:
    field = _field((False, True), (7, 6))
    dx, dy = field.grid_spacing
    # One marker in the centre of every cell, including the wrap cell between
    # the last node and the first.
    xs = (np.arange(6) + 0.5) * dx
    ys = (np.arange(6) + 0.5) * dy
    points = np.array([(x, y) for x in xs for y in ys]).T
    field.scatter(points, np.ones((1, points.shape[1])))
    data = field.field_data.local_with_halos[..., 0]
    interior = data[1:-1, :]
    # Every node of an interior column has four cell centres around it.
    assert np.allclose(interior, 1.0)
    # The bounded x edges see two cells only.
    assert np.allclose(data[0, :], 0.5)
    assert np.allclose(data[-1, :], 0.5)


def test_deposit_wraps_a_marker_beyond_the_last_node() -> None:
    field = _field((False, True), (5, 4))
    dx, dy = field.grid_spacing
    # A quarter cell past the last node along y lands 3/4 on node 3, 1/4 on node 0.
    field.scatter(np.array([[2 * dx], [3.25 * dy]]), np.array([[1.0]]))
    data = field.field_data.local_with_halos[..., 0]
    assert data[2, 3] == pytest.approx(0.75)
    assert data[2, 0] == pytest.approx(0.25)
    assert data.sum() == pytest.approx(1.0)


def test_gather_interpolates_between_the_last_and_first_node() -> None:
    field = _field((False, True), (5, 8))
    dx, dy = field.grid_spacing
    y_nodes = np.arange(8) * dy
    field.field_data.local_with_halos[..., 0] = np.cos(2.0 * np.pi * y_nodes)[None, :]
    point = np.array([[2 * dx], [7.5 * dy]])
    value = float(np.asarray(field.gather(point))[0, 0])
    expected = 0.5 * (np.cos(2.0 * np.pi * 7 * dy) + np.cos(0.0))
    assert value == pytest.approx(expected)


def test_gather_and_deposit_wrap_in_three_dimensions() -> None:
    field = _field((False, True, True), (4, 4, 4))
    dx, dy, dz = field.grid_spacing
    field.scatter(np.array([[dx], [3.5 * dy], [3.5 * dz]]), np.array([[1.0]]))
    data = field.field_data.local_with_halos[..., 0]
    assert data[1, 3, 3] == pytest.approx(0.25)
    assert data[1, 0, 0] == pytest.approx(0.25)
    assert data[1, 3, 0] == pytest.approx(0.25)
    assert data[1, 0, 3] == pytest.approx(0.25)
    value = float(
        np.asarray(field.gather(np.array([[dx], [3.5 * dy], [3.5 * dz]])))[0, 0]
    )
    assert value == pytest.approx(0.25)


def test_bounded_axes_still_clamp() -> None:
    field = _field((False, False), (5, 5))
    dx, dy = field.grid_spacing
    field.scatter(np.array([[4 * dx], [4 * dy]]), np.array([[1.0]]))
    data = field.field_data.local_with_halos[..., 0]
    assert data[4, 4] == pytest.approx(1.0)
    assert data.sum() == pytest.approx(1.0)


def test_grid_points_follow_the_periodic_layout() -> None:
    field = _field((False, True), (5, 4))
    x_points, y_points = field.get_grid_points()
    assert np.allclose(x_points, np.linspace(0.0, 1.0, 5))
    assert np.allclose(y_points, np.arange(4) * 0.25)
