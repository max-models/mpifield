"""Tests for ``Field.evaluate`` and ``Field.plot``."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import xarray as xr
from maybempi import MPI

from mpifield import Field

matplotlib.use("Agg")

# mpirun -n 2 pytest src/mpifield/tests/unit/test_field_evaluate.py --with-mpi

# plasma-plots draws on rank 0 only (other ranks get a ``SkippedPlot``), so the
# plot assertions run serially.
serial_only = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1, reason="plots are drawn on rank 0 only"
)


def _linear_field_2d(comm: MPI.Comm | None = None) -> Field:
    """A 2D field ``f = 1 + 2x - 3y``, which linear interpolation reproduces exactly."""
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(-1.0, 1.0, 9)
    data = 1.0 + 2.0 * x[:, None] - 3.0 * y[None, :]
    return Field.from_array(
        data, lower_bound=(0.0, -1.0), upper_bound=(1.0, 1.0), comm=comm
    )


def _exact(x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
    return 1.0 + 2.0 * np.asarray(x) - 3.0 * np.asarray(y)


def test_evaluate_without_coordinates_returns_the_nodes() -> None:
    """With no selection, the result is the field data on the grid nodes."""
    field = _linear_field_2d()
    array = field.evaluate(name="phi")

    assert isinstance(array, xr.DataArray)
    assert array.name == "phi"
    assert array.dims == ("x", "y")
    np.testing.assert_array_equal(array.values, np.asarray(field.to_numpy())[..., 0])
    np.testing.assert_allclose(array.x, np.linspace(0.0, 1.0, 11))
    np.testing.assert_allclose(array.y, np.linspace(-1.0, 1.0, 9))


def test_evaluate_at_nodes_reproduces_the_data() -> None:
    """Interpolating exactly at the nodes changes nothing."""
    rng = np.random.default_rng(0)
    data = rng.random((7, 5))
    field = Field.from_array(data, lower_bound=(0.0, 0.0), upper_bound=(3.0, 2.0))

    array = field.evaluate(x=np.linspace(0.0, 3.0, 7), y=np.linspace(0.0, 2.0, 5))

    np.testing.assert_allclose(array.values, data, rtol=0, atol=1e-14)


def test_scalar_drops_the_dimension_and_keeps_a_scalar_coordinate() -> None:
    field = _linear_field_2d()
    profile = field.evaluate(y=0.3)

    assert profile.dims == ("x",)
    assert float(profile.y) == 0.3
    np.testing.assert_allclose(profile.values, _exact(profile.x.values, 0.3))


def test_sequences_form_a_tensor_product_mesh() -> None:
    field = _linear_field_2d()
    xs = np.linspace(0.05, 0.95, 4)
    array = field.evaluate(x=xs, y=[0.1, -0.7])

    assert array.dims == ("x", "y")
    assert array.shape == (4, 2)
    np.testing.assert_allclose(
        array.values, _exact(xs[:, None], np.array([[0.1, -0.7]]))
    )


def test_dataarray_coordinates_evaluate_scattered_points() -> None:
    field = _linear_field_2d()
    xs = xr.DataArray([0.0, 0.33, 1.0], dims="points")
    ys = xr.DataArray([1.0, -0.2, -1.0], dims="points")

    array = field.evaluate(x=xs, y=ys)

    assert array.dims == ("points",)
    np.testing.assert_allclose(array.values, _exact(xs.values, ys.values))


def test_single_point_is_a_scalar_array() -> None:
    field = _linear_field_2d()
    assert field.evaluate(x=0.5, y=0.25).ndim == 0
    np.testing.assert_allclose(field.evaluate(x=0.5, y=0.25).item(), _exact(0.5, 0.25))


def test_bounds_are_inclusive_up_to_rounding() -> None:
    field = _linear_field_2d()
    upper = 1.0 + 1e-15
    np.testing.assert_allclose(field.evaluate(x=upper, y=1.0).item(), _exact(1.0, 1.0))


@pytest.mark.parametrize(
    ("coordinates", "error"),
    [
        ({"x": 1.5}, ValueError),
        ({"y": [-2.0, 0.0]}, ValueError),
        ({"x": np.nan}, ValueError),
        ({"x": []}, ValueError),
        ({"x": np.zeros((2, 2))}, ValueError),
        ({"z": 0.0}, ValueError),
        ({"component": 1}, IndexError),
    ],
)
def test_invalid_selections_raise(coordinates: dict, error: type) -> None:
    field = _linear_field_2d()
    with pytest.raises(error):
        field.evaluate(**coordinates)


def test_vector_field_has_a_component_dimension() -> None:
    x = np.linspace(0.0, 1.0, 5)
    data = np.stack([x, 2.0 * x, 3.0 * x], axis=-1)
    field = Field.from_array(data, lower_bound=(0.0,), upper_bound=(1.0,))

    array = field.evaluate(x=0.5)
    assert array.dims == ("component",)
    np.testing.assert_allclose(array.values, [0.5, 1.0, 1.5])

    second = field.evaluate(x=0.5, component=1)
    assert second.ndim == 0
    assert int(second.component) == 1
    np.testing.assert_allclose(second.item(), 1.0)


def test_periodic_axis_wraps_around_the_seam() -> None:
    """Between the last node and the upper bound, a periodic field interpolates to the first node."""
    data = np.array([4.0, 1.0, 2.0, 3.0])  # nodes at 0, 0.25, 0.5, 0.75
    field = Field.from_array(
        data, lower_bound=(0.0,), upper_bound=(1.0,), periodic=True
    )

    nodes = field.evaluate()
    np.testing.assert_allclose(nodes.x, [0.0, 0.25, 0.5, 0.75])

    array = field.evaluate(x=[0.75, 0.875, 1.0])
    np.testing.assert_allclose(array.values, [3.0, 3.5, 4.0])


def test_evaluate_3d() -> None:
    x, y, z = (np.linspace(0.0, 1.0, n) for n in (5, 4, 3))
    data = x[:, None, None] + 10.0 * y[None, :, None] + 100.0 * z[None, None, :]
    field = Field.from_array(
        data, lower_bound=(0.0, 0.0, 0.0), upper_bound=(1.0, 1.0, 1.0)
    )

    plane = field.evaluate(z=0.25)
    assert plane.dims == ("x", "y")
    np.testing.assert_allclose(plane.values, data[..., 0] + 25.0)


@serial_only
def test_plot_draws_lines_and_planes() -> None:
    field = _linear_field_2d()

    plane = field.plot(name="phi")
    assert plane.ax.get_title() == "phi"
    assert plane.ax.get_xlabel().strip("$") == "x"

    line = field.plot(y=0.5, name="phi")
    assert line.ax.get_title() == "phi, y = 0.5"
    assert len(line.ax.lines) == 1


@serial_only
def test_plot_into_given_axes() -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2)
    _linear_field_2d().plot(ax=axes[1], x=0.5)
    assert len(axes[1].lines) == 1
    assert len(axes[0].lines) == 0
    plt.close(fig)


@serial_only
def test_plot_3d_field_defaults_to_the_z_midplane() -> None:
    data = np.zeros((4, 4, 5))
    field = Field.from_array(
        data, lower_bound=(0.0, 0.0, -1.0), upper_bound=(1.0, 1.0, 3.0)
    )

    result = field.plot(name="n")
    assert result.ax.get_title() == "n, z = 1"


@serial_only
def test_plot_vector_field_requires_a_component() -> None:
    data = np.zeros((4, 4, 2))
    field = Field.from_array(data, lower_bound=(0.0, 0.0), upper_bound=(1.0, 1.0))
    with pytest.raises(ValueError, match="component="):
        field.plot()
    field.plot(component=1)


@serial_only
def test_plot_rejects_a_single_point() -> None:
    with pytest.raises(ValueError, match="single point"):
        _linear_field_2d().plot(x=0.5, y=0.5)


@pytest.mark.mpi(min_size=2)
def test_evaluate_gathers_a_decomposed_field() -> None:
    """Every rank gets the global result, equal to the serial one."""
    comm = MPI.COMM_WORLD
    field = _linear_field_2d(comm=comm)
    xs = np.linspace(0.0, 1.0, 13)

    array = field.evaluate(x=xs, y=0.4)

    np.testing.assert_allclose(array.values, _exact(xs, 0.4))
    assert field.evaluate().shape == (11, 9)


def test_units_label_the_result() -> None:
    field = _linear_field_2d()
    assert field.evaluate(units="V").attrs == {"units": "V"}
    assert field.evaluate().attrs == {}


@serial_only
def test_plot_shows_units_on_the_colorbar() -> None:
    result = _linear_field_2d().plot(name="phi", units="V")
    colorbar = result.ax.collections[0].colorbar
    assert colorbar.ax.get_ylabel() == "phi [V]"
