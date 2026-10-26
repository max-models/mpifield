"""Tests for field."""

from __future__ import annotations

from pathlib import Path

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI
from simgeometry.objects import Circle, Rectangle

from mpifield import Bounds, Field

BoundsLike = tuple[float, ...]


def _as_numpy(data: xp.ndarray) -> np.ndarray:
    """Return backend data as a NumPy array for assertions."""
    if hasattr(data, "get"):
        return data.get()
    return np.asarray(data)


def test_set_inside_accepts_python_scalar() -> None:
    """Verify object fills coerce Python scalars to the field dtype."""
    field = Field(
        ndim=2,
        lower_bound=(0, 0),
        upper_bound=(1, 1),
        num_gridpoints=(3, 3),
        num_components=1,
    )
    rectangle = Rectangle(midpoint=(0.5, 0.5), side_lengths=(0.25, 0.25))

    field.set_inside(rectangle, value=1.0)

    assert field.local_with_halos[1, 1, 0] == 1.0
    assert field.local_with_halos.sum() == 1.0


def test_set_inside_uses_the_node_positions_with_the_lower_bound() -> None:
    """The nodes inside a shape are found at their physical positions."""
    field = Field(
        ndim=2,
        lower_bound=(-1.0, 2.0),
        upper_bound=(1.0, 4.0),
        num_gridpoints=(21, 21),
        num_components=2,
    )
    circle = Circle(midpoint=(0.0, 3.0), radius=0.5)

    field.set_inside(circle, value=-5.0, component=1)

    x, y = np.meshgrid(*field.get_grid_points(), indexing="ij")
    inside = np.hypot(x, y - 3.0) < 0.5 - 1e-9
    data = field.local_with_halos
    assert np.all(data[..., 1][inside] == -5.0)
    assert np.all(data[..., 1][np.hypot(x, y - 3.0) > 0.5 + 1e-9] == 0.0)
    assert np.all(data[..., 0] == 0.0)


def test_linear_interpolation_matches_linear_analytic_function_1d() -> None:
    """Linear interpolation is exact for a linear analytic field."""
    field = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(1.0,),
        num_gridpoints=(9,),
        num_components=1,
    )
    grid = xp.linspace(0.0, 1.0, 9)
    field.set_field_data(2.0 + 3.0 * grid, component=0)
    positions = xp.asarray([[0.125, 0.5, 0.875]])

    interpolated = field.interpolate_field(positions)[0]

    assert xp.allclose(interpolated, 2.0 + 3.0 * positions[0])


def test_linear_scatter_matches_cloud_in_cell_weights_1d() -> None:
    """A point halfway between two nodes deposits equal CIC weights."""
    field = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(1.0,),
        num_gridpoints=(3,),
        num_components=1,
    )

    field.scatter_linear(
        positions=xp.asarray([[0.25]]),
        values=xp.asarray([[10.0]]),
    )

    assert xp.allclose(
        field.field_data.local_with_halos[:, 0], xp.asarray([5.0, 5.0, 0.0])
    )
    assert xp.sum(field.field_data.local_with_halos[:, 0]) == pytest.approx(10.0)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("periodic", [False, True])
def test_linear_scatter_of_host_marker_arrays_matches_the_general_path(
    ndim: int, periodic: bool
) -> None:
    """Float64 NumPy markers go straight to the compiled kernel, with the same deposit."""
    if xp.get_backend() != "numpy":
        pytest.skip("the direct path is for host arrays")

    def make() -> Field:
        return Field(
            ndim=ndim,
            lower_bound=(0.0,) * ndim,
            upper_bound=(1.0, 2.0, 3.0)[:ndim],
            num_gridpoints=(5, 4, 6)[:ndim],
            num_components=1,
            periodic=(False,) * (ndim - 1) + (periodic,),
        )

    rng = np.random.default_rng(3)
    num_markers = 40
    storage = (
        rng.random((ndim, num_markers + 8)) * np.array([1.0, 2.0, 3.0])[:ndim, None]
    )
    positions = storage[:, :num_markers]  # stored markers: a row-strided view
    values = rng.random((1, num_markers))
    direct, general = make(), make()

    direct.scatter_linear(positions=positions, values=values)
    # a list takes the general path, which converts and checks
    general.scatter_linear(positions=positions.tolist(), values=values.tolist())

    deposited = direct.field_data.local_with_halos
    np.testing.assert_array_equal(deposited, general.field_data.local_with_halos)
    assert deposited.sum() == pytest.approx(values.sum())


def test_linear_scatter_accumulates_repeated_indices_1d() -> None:
    """Multiple points in the same cell must all contribute to the grid."""
    field = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(1.0,),
        num_gridpoints=(3,),
        num_components=1,
    )

    field.scatter_linear(
        positions=xp.asarray([[0.25, 0.25, 0.25]]),
        values=xp.asarray([[1.0, 2.0, 3.0]]),
    )

    assert xp.allclose(
        field.field_data.local_with_halos[:, 0], xp.asarray([3.0, 3.0, 0.0])
    )


def test_linear_scatter_accepts_single_point_component_values_1d() -> None:
    """A flat value list for one point maps to field components."""
    field = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(1.0,),
        num_gridpoints=(3,),
        num_components=2,
    )

    field.scatter_linear(positions=0.25, values=[10.0, 20.0])

    assert xp.allclose(
        field.field_data.local_with_halos,
        xp.asarray([[5.0, 10.0], [5.0, 10.0], [0.0, 0.0]]),
    )


def test_gradient_matches_linear_potential_1d() -> None:
    """The numerical gradient of a linear potential is exact."""
    field = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(1.0,),
        num_gridpoints=(11,),
        num_components=1,
    )
    grid = xp.linspace(0.0, 1.0, 11)
    field.set_field_data(7.0 - 4.0 * grid, component=0)

    gradient = field.gradient()

    assert gradient.num_components == 1
    assert xp.allclose(gradient.local[:, 0], -4.0)


def test_gradient_of_a_2d_field_has_one_component_per_axis() -> None:
    field = Field(
        ndim=2,
        lower_bound=(0.0, 0.0),
        upper_bound=(1.0, 2.0),
        num_gridpoints=(11, 21),
    )
    x, y = np.meshgrid(*field.get_grid_points(), indexing="ij")
    field.set_field_data(3.0 * x - 2.0 * y)

    gradient = field.gradient()

    assert gradient.num_components == 2
    assert np.allclose(gradient.local[..., 0], 3.0)
    assert np.allclose(gradient.local[..., 1], -2.0)


def test_gradient_wraps_on_a_periodic_axis() -> None:
    """Across the seam the central difference uses the nodes on both sides."""
    n = 32
    field = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(2 * np.pi,),
        num_gridpoints=(n,),
        periodic=True,
    )
    (x,) = field.get_grid_points()
    field.set_field_data(np.sin(x))

    gradient = field.gradient().local[:, 0]

    # the central difference of sin is sin(h)/h cos(x), everywhere, seam included
    h = 2 * np.pi / n
    assert np.allclose(gradient, np.sin(h) / h * np.cos(x))


def test_gradient_needs_a_scalar_field() -> None:
    field = Field(
        ndim=2,
        lower_bound=(0.0, 0.0),
        upper_bound=(1.0, 1.0),
        num_gridpoints=4,
        num_components=2,
    )
    with pytest.raises(ValueError, match="scalar field"):
        field.gradient()


@pytest.mark.parametrize("lower_bound", [(0,), (-1,)])
@pytest.mark.parametrize("upper_bound", [(1,), (10,)])
@pytest.mark.parametrize("num_gridpoints", [64, 128, 256])
@pytest.mark.parametrize("num_components", [1, 2, 3])
def test_initialize_field_1d(
    lower_bound: BoundsLike,
    upper_bound: BoundsLike,
    num_gridpoints: int | tuple[int, ...],
    num_components: int,
    verbose: bool = False,
) -> None:
    """Verify one-dimensional field initialization."""
    ndim = 1
    field = Field(
        ndim=ndim,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_gridpoints=num_gridpoints,
        num_components=num_components,
    )

    value = 42
    index = num_gridpoints // 3
    # Test all components
    for component in range(num_components):
        field.set_value(index=index, value=value, component=component)
        assert field.get_value(index=index, component=component) == value


@pytest.mark.parametrize("lower_bound", [(0,), (-1,)])
@pytest.mark.parametrize("upper_bound", [(1,), (10,)])
@pytest.mark.parametrize("num_gridpoints", [64, 128, 256])
@pytest.mark.parametrize("num_components", [1, 2, 3])
def test_inpolate_field_1d(
    lower_bound: BoundsLike,
    upper_bound: BoundsLike,
    num_gridpoints: int | tuple[int, ...],
    num_components: int,
    verbose: bool = False,
) -> None:
    """Verify one-dimensional field interpolation."""
    ndim = 1
    field = Field(
        ndim=ndim,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_gridpoints=num_gridpoints,
        num_components=num_components,
    )

    values = [10.0] * num_components
    field.scatter_linear(positions=0.45, values=values)
    # assert xp.sum(field.field_data) == xp.sum(values)

    intp_values = field.interpolate_field(positions=0.45)
    if verbose:
        print(f"{intp_values =}")
    # assert xp.allclose(intp_values[0, 0], values)


@pytest.mark.parametrize("lower_bound", [(0, 0)])
@pytest.mark.parametrize("upper_bound", [(1, 1)])
@pytest.mark.parametrize("num_gridpoints", [(3, 3), (8, 12), (83, 123)])
@pytest.mark.parametrize("num_components", [1, 2, 3])
def test_field_magic_methods_2d(
    lower_bound: BoundsLike,
    upper_bound: BoundsLike,
    num_gridpoints: int | tuple[int, ...],
    num_components: int,
    verbose: bool = False,
) -> None:
    # Initialize fields
    """Verify two-dimensional field arithmetic operations."""
    field1 = Field(
        ndim=2,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_gridpoints=num_gridpoints,
        num_components=num_components,
    )
    field2 = field1.copy()

    # Set data in one location
    field1[0, 0, 0] = 3.0
    assert field1[0, 0, 0] == 3.0

    # Fill field1 and field2
    field1.set_field_data(data=3.0)
    field2.set_field_data(data=2.0)

    assert xp.allclose(field1.field_data.local_with_halos, 3.0)
    assert xp.allclose(field2.field_data.local_with_halos, 2.0)

    field3 = field1 + field2
    assert xp.allclose(field3.field_data.local_with_halos, 5.0)
    field4 = field1 - field2
    assert xp.allclose(field4.field_data.local_with_halos, 1.0)
    field5 = field1 * field2
    assert xp.allclose(field5.field_data.local_with_halos, 6.0)
    field6 = field1 / field2
    assert xp.allclose(field6.field_data.local_with_halos, 1.5)
    field7 = -field1
    assert xp.allclose(field7.field_data.local_with_halos, -3.0)

    field8 = 2.0 + field1
    assert xp.allclose(field8.field_data.local_with_halos, 5.0)
    field9 = 10.0 - field1
    assert xp.allclose(field9.field_data.local_with_halos, 7.0)
    field1 += 1.0
    assert xp.allclose(field1.field_data.local_with_halos, 4.0)


def test_field_darray_global_data_api() -> None:
    """Verify Field delegates global data operations to DistributedArray."""
    # the decomposed axis needs at least one cell per rank
    size = MPI.COMM_WORLD.Get_size()
    data = xp.arange(4 * 3 * size, dtype=float).reshape(3 * size, 4)
    field = Field.from_array(
        data,
        lower_bound=(0.0, 0.0),
        upper_bound=(1.0, 1.0),
        comm=MPI.COMM_WORLD,
        domain_decomposition=[True, False],
        num_ghost_points=1,
    )

    assert field.num_components == 1
    assert xp.allclose(field.to_ndarray()[..., 0], data)
    numpy_data = field.to_numpy()
    assert isinstance(numpy_data, np.ndarray)
    assert np.allclose(numpy_data[..., 0], _as_numpy(data))
    assert xp.allclose(field.get_field_data(component=0, gathered=True), data)
    assert field.local.shape[-1] == 1
    assert field.local_with_halos.shape == field.field_data.local_with_halos.shape

    replacement = data + 10.0
    field.set_field_data(replacement, component=0)
    assert xp.allclose(field.to_ndarray()[..., 0], replacement)

    copied = field.copy()
    assert copied is not field
    assert xp.allclose(copied.to_ndarray(), field.to_ndarray())

    as_int = field.astype(int)
    assert as_int.dtype == xp.dtype(int)
    assert xp.allclose(as_int.to_ndarray(), replacement[..., None].astype(int))

    assert xp.allclose(field.sum(component=0), xp.sum(replacement))
    assert xp.allclose(field.mean(component=0), xp.mean(replacement))
    assert xp.allclose(field.max(component=0), xp.max(replacement))

    mask = field > 15.0
    assert mask.dtype == xp.dtype(bool)
    assert xp.allclose(mask.to_ndarray()[..., 0], replacement > 15.0)


def test_field_vector_operations() -> None:
    """Verify vector field helpers."""
    # the decomposed axis needs at least one cell per rank
    shape = (max(4, MPI.COMM_WORLD.Get_size()), 3, 3)
    x_field = Field.from_array(
        xp.zeros(shape, dtype=float),
        lower_bound=(0.0, 0.0),
        upper_bound=(1.0, 1.0),
        comm=MPI.COMM_WORLD,
        domain_decomposition=[True, False],
    )
    y_field = x_field.copy()
    x_field.set_field_data(1.0, component=0)
    y_field.set_field_data(1.0, component=1)

    cross = x_field.cross(y_field)
    assert xp.allclose(cross.get_field_data(component=0), 0.0)
    assert xp.allclose(cross.get_field_data(component=1), 0.0)
    assert xp.allclose(cross.get_field_data(component=2), 1.0)

    magnitude = cross.magnitude()
    assert magnitude.num_components == 1
    assert xp.allclose(magnitude.get_field_data(component=0), 1.0)

    dot = x_field.dot(y_field)
    assert dot.num_components == 1
    assert xp.allclose(dot.get_field_data(component=0), 0.0)


@pytest.mark.parametrize("filter_name", ["binomial", "gaussian", "uniform"])
def test_field_smoothing_filters_reduce_peak(filter_name: str) -> None:
    """Smoothing filters spread an impulse while preserving periodic total mass."""
    field = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(1.0,),
        num_gridpoints=(33,),
        num_components=1,
        periodic=True,
    )
    impulse = xp.zeros((33,), dtype=float)
    impulse[16] = 1.0
    field.set_field_data(impulse, component=0)

    smoothed = field.smooth(filter_name=filter_name, passes=1, in_place=False)
    original = field.get_field_data(component=0, gathered=True)
    filtered = smoothed.get_field_data(component=0, gathered=True)

    assert filtered.shape == original.shape
    assert float(xp.max(filtered)) < float(xp.max(original))
    assert float(xp.sum(filtered)) == pytest.approx(float(xp.sum(original)))


def test_field_smoothing_single_component_only() -> None:
    """Component-selective smoothing leaves other components unchanged."""
    field = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(1.0,),
        num_gridpoints=(9,),
        num_components=2,
        periodic=True,
    )
    comp0 = xp.zeros((9,), dtype=float)
    comp0[4] = 1.0
    comp1 = xp.linspace(0.0, 1.0, 9)
    field.set_field_data(comp0, component=0)
    field.set_field_data(comp1, component=1)

    field.smooth_binomial(component=0, passes=1, in_place=True)

    assert float(xp.max(field.get_field_data(component=0, gathered=True))) < 1.0
    assert xp.allclose(field.get_field_data(component=1, gathered=True), comp1)


@pytest.mark.parametrize("lower_bound", [(0, 0), (-1, -1)])
@pytest.mark.parametrize("upper_bound", [(1, 1), (10, 10)])
@pytest.mark.parametrize("num_gridpoints", [(3, 3), (8, 12), (83, 123)])
def test_field_io(
    lower_bound: BoundsLike,
    upper_bound: BoundsLike,
    num_gridpoints: int | tuple[int, ...],
    tmp_path: Path,
    verbose: bool = False,
) -> None:
    """Verify field export and loading behavior."""
    ndim = len(num_gridpoints)

    # Initialize fields
    field1 = Field(
        ndim=ndim,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_gridpoints=num_gridpoints,
        num_components=3,
    )
    field1.set_field_data(data=3.0)

    # Per-process directory, so concurrent MPI ranks never share the file.
    file_path = tmp_path / "field1.nc"
    field1.export(file_path=file_path)
    field2 = Field.load(file_path=file_path)
    assert field2.num_components == 3
    assert field2.num_gridpoints == field1.num_gridpoints
    assert np.allclose(field2.lower_bound, lower_bound)
    assert np.allclose(field2.upper_bound, upper_bound)
    assert xp.allclose(field2.local_with_halos, 3.0)


@pytest.mark.parametrize("num_components", [1, 2])
def test_export_and_load_keep_the_values_and_the_periodic_axes(
    num_components: int, tmp_path: Path
) -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(6, 5, num_components))
    field = Field.from_array(
        data if num_components > 1 else data[..., 0],
        lower_bound=(0.0, -1.0),
        upper_bound=(1.0, 1.0),
        periodic=(True, False),
    )
    field.export(tmp_path / "field.nc")

    loaded = Field.load(tmp_path / "field.nc")

    assert loaded.num_components == num_components
    assert tuple(bool(p) for p in loaded.periodic) == (True, False)
    np.testing.assert_array_equal(loaded.to_numpy(), field.to_numpy())


def test_1d_gather_uses_the_edge_cell_outside_the_grid() -> None:
    """Markers beyond the grid get the edge node values, as in the compiled pusher."""
    field = np.array([[1.0], [2.0], [4.0], [8.0]])
    bounds = Bounds(lower_bound=np.array([0.0]), upper_bound=np.array([3.0]))
    points = np.array([[-0.5, 0.5, 2.5, 3.0, 3.7]])
    result = np.zeros((1, 5))
    Field.interpolate_field_1d(
        points, field, bounds, np.array([1.0]), np.array([4]), result
    )
    np.testing.assert_allclose(result[0], [1.0, 1.5, 6.0, 8.0, 8.0])


@pytest.mark.parametrize("ndim", [2, 3])
def test_stored_block_with_halo_nodes_deposits_and_interpolates_as_the_whole_grid(
    ndim: int,
) -> None:
    """With halo nodes the kernels see the stored block as a grid of its own.

    A periodic first axis: what the halo nodes collect is added across the seam
    (``accumulate_halos``), and after ``update_halos`` the interpolation reads
    the wrapped nodes there, as the periodic kernels do on the bare grid.
    """
    shape = (7, 5, 4)[:ndim]
    periodic = (True,) + (False,) * (ndim - 1)
    kwargs = {
        "ndim": ndim,
        "lower_bound": (0.0,) * ndim,
        "upper_bound": (1.0,) * ndim,
        "num_gridpoints": shape,
        "periodic": periodic,
        "comm": MPI.COMM_SELF,
    }
    bare = Field(**kwargs)
    padded = Field(**kwargs, num_ghost_points=1)
    stored = padded.block_grid(cpu=True)
    assert tuple(stored.num_gridpoints) == tuple(n + 2 for n in shape)
    assert stored.periodic == (False,) * ndim

    rng = np.random.default_rng(5)
    positions = rng.uniform(0.0, 1.0, (ndim, 400))
    positions[1:] *= 0.999  # inside the bounded axes
    values = rng.uniform(0.5, 1.5, 400)
    bare.scatter_linear(positions, values)
    padded.scatter_linear(positions, values)
    padded.field_data.accumulate_halos()
    assert np.allclose(padded.field_data.local, bare.field_data.local)

    padded.field_data.update_halos()
    assert np.allclose(
        padded.interpolate_field(positions), bare.interpolate_field(positions)
    )


class _Potential(Field):
    """A subclass, to check that derived fields keep the type."""


def test_derived_fields_keep_the_subclass() -> None:
    field = _Potential(
        ndim=2, lower_bound=(0.0, 0.0), upper_bound=(1.0, 1.0), num_gridpoints=5
    )
    field.set_field_data(2.0)

    for derived in (
        field.copy(),
        field + 1.0,
        -field,
        np.sqrt(field),
        field.gradient(),
        field.gradient().magnitude(),
        field.astype(np.float32),
    ):
        assert type(derived) is _Potential


def test_arithmetic_with_a_field_on_another_grid_raises() -> None:
    field = Field(ndim=1, lower_bound=(0.0,), upper_bound=(1.0,), num_gridpoints=5)
    other = Field(ndim=1, lower_bound=(0.0,), upper_bound=(2.0,), num_gridpoints=5)
    with pytest.raises(ValueError, match="upper_bound"):
        field.dot(other)
    with pytest.raises(TypeError, match="expected a Field"):
        field._check_field_compatibility(np.zeros(5))  # type: ignore[arg-type]


def test_block_grid_of_an_undivided_field_is_the_grid() -> None:
    field = Field(
        ndim=2,
        lower_bound=(-1.0, 0.0),
        upper_bound=(1.0, 3.0),
        num_gridpoints=(5, 7),
        periodic=(True, False),
    )
    grid = field.block_grid(cpu=True)

    np.testing.assert_allclose(grid.bounds.lower_bound, [-1.0, 0.0])
    np.testing.assert_allclose(grid.grid_spacing, [0.4, 0.5])  # periodic: n cells
    assert tuple(grid.num_gridpoints) == (5, 7)
    assert grid.periodic == (True, False)
    assert field.block_grid(cpu=True) is grid  # cached


def test_set_field_data_rejects_a_wrong_shape_and_component() -> None:
    field = Field(ndim=1, lower_bound=(0.0,), upper_bound=(1.0,), num_gridpoints=5)
    with pytest.raises(ValueError, match="Expected data of shape"):
        field.set_field_data(np.zeros(7))
    with pytest.raises(IndexError):
        field.set_field_data(1.0, component=1)


def test_repr_names_the_grid() -> None:
    field = Field(
        ndim=2, lower_bound=(0.0, 0.0), upper_bound=(1.0, 1.0), num_gridpoints=3
    )
    assert repr(field).startswith("Field(ndim=2, num_gridpoints=(3, 3)")
    assert "stored block (3, 3, 1)" in str(field)


def test_bounds_is_a_named_pair() -> None:
    bounds = Bounds(np.zeros(2), np.ones(2))
    assert bounds.lower_bound is bounds[0]


def test_a_scalar_field_broadcasts_over_the_components() -> None:
    kwargs = {"ndim": 2, "lower_bound": (0.0, 0.0), "upper_bound": (1.0, 1.0)}
    vector = Field(num_gridpoints=4, num_components=3, **kwargs)
    vector.set_field_data(np.arange(4 * 4 * 3, dtype=float).reshape(4, 4, 3))
    scale = Field(num_gridpoints=4, **kwargs)
    scale.set_field_data(np.full((4, 4), 2.0))

    for result in (vector / scale, scale * vector, vector - scale, scale - vector):
        assert result.num_components == 3
    np.testing.assert_allclose((vector / scale).local, vector.local / 2.0)
    np.testing.assert_allclose((scale - vector).local, 2.0 - vector.local)

    vector *= scale
    np.testing.assert_allclose(
        vector.local[..., 1], 2.0 * (3 * np.arange(16) + 1).reshape(4, 4)
    )

    two = Field(num_gridpoints=4, num_components=2, **kwargs)
    with pytest.raises(ValueError, match="cannot combine"):
        vector + two
    with pytest.raises(ValueError, match="cannot update"):
        scale += vector
