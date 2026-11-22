"""Faces and cells of the spline sequence: `Splines.faces`, `Splines.cells`."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from mpifield import BlockGrid, splines
from mpifield.kernels.catalog import KERNELS
from mpifield.kernels.spline_forms import (
    gather_forms_spline,
    num_components,
    scatter_forms_spline,
    staggered,
)

SHAPE = (9, 8, 7)
SPACING = np.array([0.5, 0.4, 0.3])


def _grid(ndim, periodic):
    return BlockGrid.from_geometry(
        np.zeros(ndim), SPACING[:ndim], SHAPE[:ndim], (periodic,) * ndim
    )


def _next(values, axis, periodic):
    """The values at ``k + 1`` along ``axis``; zero past the last node of a bounded axis."""
    shifted = np.roll(values, -1, axis=axis)
    if not periodic:
        index = [slice(None)] * values.ndim
        index[axis] = -1
        shifted[tuple(index)] = 0.0
    return shifted


def _edges(rng, periodic):
    """A random edge field in 3D, zero on the edges that do not exist."""
    edges = rng.normal(size=(*SHAPE, 3))
    if not periodic:
        for axis in range(3):
            index = [slice(None)] * 3
            index[axis] = -1
            edges[(*index, axis)] = 0.0
    return edges


def _curl(edges, periodic):
    """The circulation of ``edges`` around every face (the discrete curl C)."""
    ex, ey, ez = edges[..., 0], edges[..., 1], edges[..., 2]
    faces = np.empty_like(edges)
    faces[..., 0] = ey + _next(ez, 1, periodic) - _next(ey, 2, periodic) - ez
    faces[..., 1] = ez + _next(ex, 2, periodic) - _next(ez, 0, periodic) - ex
    faces[..., 2] = ex + _next(ey, 0, periodic) - _next(ex, 1, periodic) - ey
    return faces


def _derivatives(function, points, step=1e-6):
    """``d function_i / d x_j`` at ``points`` (3, N) by central differences: [j][i]."""
    rows = []
    for axis in range(points.shape[0]):
        shift = np.zeros((points.shape[0], 1))
        shift[axis] = step
        rows.append((function(points + shift) - function(points - shift)) / (2 * step))
    return rows


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_the_faces_of_the_curl_gather_the_curl_of_the_edge_field(periodic, degree):
    rng = np.random.default_rng(degree)
    grid = _grid(3, periodic)
    shape = splines(degree)
    edges = _edges(rng, periodic)
    faces = _curl(edges, periodic)
    area = np.array(
        [SPACING[1] * SPACING[2], SPACING[0] * SPACING[2], SPACING[0] * SPACING[1]]
    )
    # inside, off the knot planes where degree 1 has kinks
    points = np.ascontiguousarray(rng.uniform(1.13, 4.87, (60, 3)).T * SPACING[:, None])
    field = np.zeros((3, points.shape[1]))
    shape.faces.gather_block(points, faces, field, grid)

    def edge_field(x):
        result = np.zeros((3, x.shape[1]))
        shape.edges.gather_block(x, edges, result, grid)
        return result

    d = _derivatives(edge_field, points)
    curl = np.array([d[1][2] - d[2][1], d[2][0] - d[0][2], d[0][1] - d[1][0]])
    np.testing.assert_allclose(field / area[:, None], curl, rtol=0, atol=2e-6)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("degree", [2, 3])
def test_the_cells_of_the_divergence_gather_the_divergence_of_the_face_field(
    periodic, degree
):
    rng = np.random.default_rng(10 + degree)
    grid = _grid(3, periodic)
    shape = splines(degree)
    faces = rng.normal(size=(*SHAPE, 3))
    if not periodic:  # faces past the last node across their normal do not exist
        for normal in range(3):
            for axis in range(3):
                if axis != normal:
                    index = [slice(None)] * 3
                    index[axis] = -1
                    faces[(*index, normal)] = 0.0
    cells = sum(
        _next(faces[..., axis], axis, periodic) - faces[..., axis] for axis in range(3)
    )
    if not periodic:
        for axis in range(3):
            index = [slice(None)] * 3
            index[axis] = -1
            cells[tuple(index)] = 0.0
    area = np.array(
        [SPACING[1] * SPACING[2], SPACING[0] * SPACING[2], SPACING[0] * SPACING[1]]
    )
    # anywhere in the grid, walls included
    upper = SPACING * (np.array(SHAPE) - 1)
    points = np.ascontiguousarray(
        rng.uniform(1e-3, 1.0, (60, 3)).T * (upper[:, None] - 2e-3)
    )
    density = np.zeros((1, points.shape[1]))
    shape.cells.gather_block(points, cells[..., None], density, grid)

    def face_field(x):
        result = np.zeros((3, x.shape[1]))
        shape.faces.gather_block(x, faces, result, grid)
        return result / area[:, None]

    d = _derivatives(face_field, points, step=1e-7)
    divergence = d[0][0] + d[1][1] + d[2][2]
    np.testing.assert_allclose(
        density[0] / np.prod(SPACING), divergence, rtol=0, atol=1e-5
    )


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(("ndim", "periodic"), [(1, False), (2, True), (3, False)])
def test_cells_and_faces_are_adjoint_and_compiled_like_numpy(degree, ndim, periodic):
    rng = np.random.default_rng(degree + 3 * ndim)
    grid = _grid(ndim, periodic)
    upper = SPACING[:ndim] * (np.array(SHAPE[:ndim]) - 1)
    points = np.ascontiguousarray(
        rng.uniform(-0.5, 1.0, (300, ndim)).T * (upper[:, None] + 0.5)
    )
    for form in range(1, ndim + 1):
        components = num_components(form, ndim)
        values = rng.normal(size=(components, points.shape[1]))
        values[:, ::7] = 0.0
        field = rng.normal(size=(*SHAPE[:ndim], components))
        found = {}
        for implementation in ("pyccel", "numpy"):
            with xp.kernels.use_host_kernel_implementation(implementation):
                deposit = np.zeros_like(field)
                KERNELS[f"scatter_forms_spline_{ndim}d"](
                    points, values, deposit, *grid.geometry, degree, form
                )
                gathered = np.zeros_like(values)
                KERNELS[f"gather_forms_spline_{ndim}d"](
                    points, field, gathered, *grid.geometry, degree, form
                )
            found[implementation] = (deposit, gathered)
        for compiled, reference in zip(found["pyccel"], found["numpy"], strict=True):
            np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-12)
        deposit, gathered = found["numpy"]
        assert np.sum(gathered * values) == pytest.approx(np.sum(field * deposit))


def test_cells_sum_to_one_inside_and_hold_the_whole_deposit_when_periodic():
    grid = _grid(2, True)
    points = np.random.default_rng(1).uniform(0.0, 3.0, (2, 200))
    for degree in (1, 2, 3):
        cells = splines(degree).cells
        deposit = np.zeros((*SHAPE[:2], 1))
        cells.scatter_block(points, np.ones(200), deposit, grid)
        assert deposit.sum() == pytest.approx(200.0)
        gathered = np.zeros((1, 200))
        cells.gather_block(points, np.ones((*SHAPE[:2], 1)), gathered, grid)
        np.testing.assert_allclose(gathered, 1.0)


def test_staggering_and_components_of_the_forms():
    assert [staggered(1, 3, 0, axis) for axis in range(3)] == [True, False, False]
    assert [staggered(2, 3, 0, axis) for axis in range(3)] == [False, True, True]
    assert all(staggered(2, 2, 0, axis) for axis in range(2))
    assert not staggered(0, 2, 0, 0)
    assert [num_components(form, 3) for form in (1, 2, 3)] == [3, 3, 1]
    assert splines(1).faces.gather.keys() == {3}
    with pytest.raises(ValueError, match="forms of 2 dimensions are 1 to 2"):
        gather_forms_spline(
            np.zeros((2, 1)),
            np.zeros((3, 3, 1)),
            np.zeros((1, 1)),
            [0.0, 0.0],
            [1.0, 1.0],
            [3, 3],
            [False, False],
            2,
            3,
        )
    with pytest.raises(ValueError, match="has 3 components, got 1"):
        scatter_forms_spline(
            np.zeros((3, 1)),
            np.ones((1, 1)),
            np.zeros((3, 3, 3, 1)),
            [0.0] * 3,
            [1.0] * 3,
            [3] * 3,
            [False] * 3,
            2,
            1,
        )
