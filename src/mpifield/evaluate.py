"""Evaluate a ``Field`` as a labelled ``xarray.DataArray``, and plot it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cunumpy as xp
import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from mpifield.field import Field

AXES = ("x", "y", "z")

# Relative tolerance on the domain bounds, so that coordinates computed in
# floating point (e.g. ``np.linspace(lower, upper, n)``) are not rejected.
BOUNDS_RTOL = 1e-12


def _check_coordinate(
    axis: str, value: Any, lower: float, upper: float
) -> float | np.ndarray | xr.DataArray:
    """Validate one axis selection and return it as ``float`` data."""
    import xarray as xr

    if isinstance(value, xr.DataArray):
        values = value.astype(float)
    else:
        values = np.asarray(value, dtype=float)
        if values.ndim > 1:
            raise ValueError(
                f"{axis} must be a scalar or a one-dimensional coordinate sequence; "
                "pass xarray.DataArray coordinates sharing a dimension for scattered points"
            )
        if values.ndim == 1 and not values.size:
            raise ValueError(f"{axis} must contain at least one coordinate")

    data = np.asarray(values)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{axis} values must be finite")
    tol = BOUNDS_RTOL * max(abs(lower), abs(upper), upper - lower)
    if np.any((data < lower - tol) | (data > upper + tol)):
        raise ValueError(
            f"{axis} values must lie in the domain [{lower}, {upper}], "
            f"got values in [{data.min()}, {data.max()}]"
        )
    values = values.clip(lower, upper)
    return float(values) if np.ndim(values) == 0 else values


def evaluate_field(
    field: Field,
    coordinates: dict[str, Any],
    component: int | None = None,
    name: str | None = None,
    units: str | None = None,
) -> xr.DataArray:
    """Evaluate ``field`` on the requested coordinates; see ``Field.evaluate``."""
    import xarray as xr

    axes = AXES[: field.ndim]
    for axis, value in coordinates.items():
        if value is not None and axis not in axes:
            raise ValueError(
                f"{axis!r} is not an axis of a {field.ndim}D field; its axes are {axes}"
            )
    if component is not None and not 0 <= component < field.num_components:
        raise IndexError(
            f"component {component} is out of range for a field with "
            f"{field.num_components} components"
        )

    # Collective under MPI: every rank gathers the global field.
    data = np.asarray(field.to_numpy())
    grid_points = [
        np.asarray(xp.to_numpy(points), dtype=float)
        for points in field.get_grid_points()
    ]
    lower = [float(bound) for bound in field.lower_bound]
    upper = [float(bound) for bound in field.upper_bound]

    if component is not None:
        data = data[..., component]
    elif field.num_components == 1:
        data = data[..., 0]
    dims = axes + (("component",) if data.ndim > field.ndim else ())
    coords: dict[str, Any] = dict(zip(axes, grid_points, strict=True))
    if "component" in dims:
        coords["component"] = np.arange(field.num_components)
    elif component is not None:
        coords["component"] = component
    attrs = {"units": units} if units is not None else {}
    nodes = xr.DataArray(data, dims=dims, coords=coords, name=name, attrs=attrs)

    selection = {
        axis: _check_coordinate(axis, value, lower[dim], upper[dim])
        for dim, (axis, value) in enumerate(
            zip(axes, (coordinates.get(a) for a in axes), strict=True)
        )
        if value is not None
    }
    if not selection:
        return nodes

    # A periodic axis stores n nodes over its length; repeat the first node at
    # the upper bound so that interpolation across the seam wraps around.
    periodic = field.periodic or (False,) * field.ndim
    if isinstance(periodic, bool):
        periodic = (periodic,) * field.ndim
    for dim, axis in enumerate(axes):
        if periodic[dim] and axis in selection:
            seam = nodes.isel({axis: [0]}).assign_coords({axis: [upper[dim]]})
            nodes = xr.concat([nodes, seam], dim=axis)

    return nodes.interp(selection, method="linear", assume_sorted=True)


def _title(array: xr.DataArray) -> str | None:
    """Name the plot after the array and the coordinates fixed by the selection."""
    fixed = [
        f"{dim} = {float(array.coords[dim]):.4g}"
        for dim in AXES
        if dim in array.coords and dim not in array.dims
    ]
    parts = ([array.name] if array.name else []) + fixed
    return ", ".join(parts) or None


def plot_field(
    field: Field,
    coordinates: dict[str, Any],
    component: int | None = None,
    name: str | None = None,
    units: str | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot ``field`` with plasma-plots; see ``Field.plot``."""
    import matplotlib.pyplot as plt
    import plasma_plots  # noqa: F401  (registers the ``.plasma`` accessor)

    # Resolve the pyplot backend before plasma-plots draws inside
    # ``plt.rc_context``: if the first figure of a Jupyter kernel is created in
    # there, leaving the context resets the interactive flag that IPython just
    # set, and later figures are no longer shown inline.
    # TODO: remove once mpifield requires a plasma-plots release that keeps the
    # interactive state itself (fixed after 0.1.1).
    plt.switch_backend(plt.get_backend())

    if component is None and field.num_components > 1:
        raise ValueError(
            f"the field has {field.num_components} components; choose one with "
            "component=, or plot field.magnitude()"
        )

    array = evaluate_field(
        field, coordinates, component=component, name=name, units=units
    )
    if array.ndim == 3:
        # Default to the midplane of the domain in z.
        midplane = 0.5 * (float(field.lower_bound[2]) + float(field.upper_bound[2]))
        array = evaluate_field(
            field,
            {**coordinates, "z": midplane},
            component=component,
            name=name,
            units=units,
        )

    spatial = [dim for dim in array.dims if dim in AXES]
    if len(spatial) != array.ndim:
        raise ValueError(
            f"cannot plot an evaluation with dimensions {array.dims}; "
            "plot() needs a line or a plane of x, y, z coordinates"
        )
    kwargs.setdefault("title", _title(array))
    if array.ndim == 1:
        return array.plasma.plot.lineout(x=spatial[0], ax=ax, **kwargs)
    if array.ndim == 2:
        return array.plasma.plot.slice(x=spatial[0], y=spatial[1], ax=ax, **kwargs)
    raise ValueError(
        "cannot plot a single point; select at most one coordinate per axis"
    )
