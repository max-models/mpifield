"""A field on the nodes of a uniform grid, split over MPI ranks, with gather and scatter.

:class:`Field` is a :class:`simgeometry.grid.Grid` holding an
:class:`mpiarray.DistributedArray` of shape ``num_gridpoints + (num_components,)``:
the grid axes are split over the ranks' process grid with ``num_ghost_points`` halo
nodes, the component axis is never split. On it:

- :meth:`Field.gather` interpolates the field at marker positions and
  :meth:`Field.scatter` deposits marker values onto the nodes, both with a
  :class:`~mpifield.weighting.Weighting` (linear cloud-in-cell by default, or
  quadratic, or one of your own) through compiled kernels (Pyccel on the CPU, CUDA
  for CuPy arrays, a NumPy fallback);
- elementwise arithmetic and NumPy ufuncs return new fields; ``sum``, ``min``, ...
  reduce over all ranks;
- :meth:`Field.evaluate` returns the field, or a line or plane of it, as an
  :class:`xarray.DataArray`, and :meth:`Field.plot` draws it with plasma-plots.

Positions are component-major, ``(ndim, N)``, and per-marker values ``(ncomp, N)``,
as the kernels take them.
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Any, Literal, cast

import cunumpy as xp
import mpiarray as mpa
import numpy as np
import scope_profiler as sp
from maybempi import MPI
from mpiarray import DistributedArray
from simgeometry.grid import Grid

from mpifield._profiling import PROFILE_TAGS
from mpifield._types import Array, BlockGrid, Bounds, IndexLike, PathLikeStr
from mpifield.weighting import LINEAR, Weighting

if TYPE_CHECKING:
    import xarray as xr
    from simgeometry.domain_decomposition import DomainDecomposition
    from simgeometry.shapes import Shape
    from typing_extensions import Self

__all__ = ["BlockGrid", "Field"]


def _periodic_flags(periodic: object, ndim: int) -> tuple[bool, ...]:
    """Normalise a ``periodic`` argument (None, a bool or a sequence) to one flag per axis."""
    if periodic is None:
        return (False,) * ndim
    if isinstance(periodic, bool):
        return (periodic,) * ndim
    flags = tuple(bool(flag) for flag in periodic)  # type: ignore[union-attr]
    return (
        flags[:ndim] if len(flags) >= ndim else flags + (False,) * (ndim - len(flags))
    )


class Field(Grid):
    """A field on the nodes of a uniform grid, split over MPI ranks.

    ``num_gridpoints`` nodes (one count per axis, or one for all) span
    ``lower_bound`` to ``upper_bound`` on a bounded axis;
    on a periodic axis the last node neighbours the first, so there are
    ``num_gridpoints`` cells over the length. ``comm`` and ``domain_decomposition``
    split the grid axes over the ranks as :class:`simgeometry.grid.Grid` does; each
    rank stores its block with ``num_ghost_points`` halo nodes on every side
    (:attr:`local_with_halos`). Without ``comm`` every rank holds the whole field.

    Examples:
    --------
    >>> phi = Field(ndim=2, lower_bound=(0, 0), upper_bound=(1, 1), num_gridpoints=33)
    >>> phi.set_field_data(np.zeros((33, 33)))
    >>> values = phi.gather(np.array([[0.5], [0.25]]))  # (1, 1)
    """

    __array_priority__ = 1000

    # Layout ------------------------------------------------------------------#

    def _periodic_for_darray(self) -> tuple[bool, ...] | None:
        """Return periodic flags for the grid axes plus the component axis."""
        if self.periodic is None:
            return None
        return _periodic_flags(self.periodic, self.ndim) + (False,)

    @staticmethod
    def _decomposition_of(grid: Grid) -> DomainDecomposition:
        """The domain decomposition of ``grid`` (simgeometry types it loosely)."""
        return cast("DomainDecomposition", grid.domain_decomposition_class)

    def _layout(
        self,
        num_components: int,
        periodic: tuple[bool, ...] | None = None,
        halo: int | None = None,
    ) -> mpa.Layout:
        """Return the layout of a ``num_gridpoints + (num_components,)`` array.

        The grid axes are split over the process grid and carry ``halo`` ghost
        nodes (default: the field's); the component axis stays on every rank.
        """
        halo = self.num_ghost_points if halo is None else halo
        return self._decomposition_of(self).layout(
            self.num_gridpoints + (num_components,),
            halo=(halo,) * self.ndim + (0,),
            periodic=periodic,
        )

    def __init__(
        self,
        ndim: int,
        lower_bound: tuple,
        upper_bound: tuple,
        num_gridpoints: int | tuple[int, ...],
        num_components: int = 1,
        comm: MPI.Comm | None = None,
        num_ghost_points: int = 0,
        domain_decomposition: Any = None,
        periodic: bool | tuple[bool, ...] | None = None,
        dtype: type = float,
    ) -> None:
        """Create a zero field; see the class docstring for the arguments."""
        if isinstance(num_gridpoints, (int, np.integer)):
            num_gridpoints = (int(num_gridpoints),) * ndim
        super().__init__(
            ndim=ndim,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            num_gridpoints=num_gridpoints,
            num_ghost_points=num_ghost_points,
            comm=comm,
            domain_decomposition=domain_decomposition,
            periodic=periodic,
        )
        if num_components < 1:
            raise ValueError(f"num_components must be >= 1, got {num_components}")
        self._num_components = num_components
        self._dtype = xp.dtype(dtype)
        # only the grid axes are split; the component axis stays on every rank
        self._field_data = mpa.zeros(
            dtype=self.dtype,
            layout=self._layout(num_components, self._periodic_for_darray()),
        )

    def _make_block_grid(
        self, layout: mpa.Layout, cpu: bool, inside_domain: bool
    ) -> BlockGrid:
        """Compute :meth:`block_grid`."""
        ndim = self.ndim
        halo = layout.halo[:ndim]
        flags = _periodic_flags(self.periodic, ndim)
        first, last = [], []  # the stored nodes the grid starts and ends with
        for axis in range(ndim):
            start = layout.index_bounds[axis][0]
            owned = layout.local_shape[axis]
            h = halo[axis]
            # the halo nodes beyond a bounded edge of the domain are outside it
            trim = inside_domain and not flags[axis]
            low = h if trim and start == 0 else 0
            high = h if trim and start + owned == self.num_gridpoints[axis] else 0
            first.append(low)
            last.append(owned + 2 * h - high)
        spacing = np.asarray(xp.to_numpy(self.grid_spacing), dtype=float)
        start = np.array(
            [
                bounds[0] - h + low
                for bounds, h, low in zip(
                    layout.index_bounds[:ndim], halo, first, strict=True
                )
            ],
            dtype=float,
        )
        num_gridpoints = np.array(
            [stop - low for low, stop in zip(first, last, strict=True)], dtype=int
        )
        lower = np.asarray(xp.to_numpy(self.bounds.lower_bound), dtype=float)
        lower = lower + start * spacing
        upper = lower + (num_gridpoints - 1) * spacing
        periodic = tuple(
            bool(flag) and h == 0 for flag, h in zip(flags, halo, strict=True)
        )
        convert = np.asarray if cpu else xp.asarray
        return BlockGrid(
            bounds=Bounds(lower_bound=convert(lower), upper_bound=convert(upper)),
            grid_spacing=convert(spacing),
            num_gridpoints=num_gridpoints if cpu else xp.asarray(num_gridpoints),
            periodic=periodic,
            geometry=(
                *(float(v) for v in lower),
                *(float(v) for v in spacing),
                *(int(n) for n in num_gridpoints),
                *(int(flag) for flag in periodic),
            ),
            index=tuple(
                slice(low, stop) for low, stop in zip(first, last, strict=True)
            ),
        )

    def block_grid(self, cpu: bool = False, inside_domain: bool = False) -> BlockGrid:
        """The grid of this rank's stored block (halo nodes included), for the kernels.

        Under a spatial decomposition the kernels deposit into and interpolate
        from :attr:`local_with_halos`, a block of the global grid that starts
        ``halo`` nodes before the block's first node. They see it as a grid of
        its own: its lower bound is that node, its node count the stored extent.
        A periodic axis is periodic for the kernels only without halo nodes; with
        them the halo nodes hold the wrapped-around nodes (:meth:`fill_halos`,
        :meth:`exchange_halos`). ``cpu`` returns host arrays, for the NumPy and
        Pyccel kernels. Cached per layout.

        With ``inside_domain`` the grid leaves out the halo nodes beyond a bounded
        edge of the domain, which no exchange fills or collects: the kernels then
        clamp there as on an unsplit field, and a stencil wider than a cell
        neither deposits into nor reads from nodes outside the domain. This is the
        grid of :meth:`scatter` and :meth:`gather`. ``local_with_halos[grid.index]``
        is the part of the stored block it describes; without ``inside_domain``
        that is all of it.
        """
        layout = self.field_data.layout
        cache = self.__dict__.setdefault("_block_grids", {})
        key = (layout, cpu, inside_domain)
        if key not in cache:
            if len(cache) > 4:
                cache.clear()
            cache[key] = self._make_block_grid(layout, cpu, inside_domain)
        return cache[key]

    # Construction ------------------------------------------------------------#

    @classmethod
    def on_grid(cls, grid: Grid, num_components: int = 1, split: bool = True) -> Self:
        """Return a zero field on the nodes of ``grid``, split as ``grid`` is.

        On a spatially decomposed grid each rank holds its own nodes plus one
        halo node on every side, which the deposit accumulates into and the
        interpolation reads from (:meth:`block_grid`); otherwise, or with
        ``split=False``, every rank holds the whole grid.
        """
        decomposition = cls._decomposition_of(grid)
        distributed = (
            split
            and not decomposition.replicated
            and math.prod(decomposition.proc_sizes) > 1
        )
        return cls(
            ndim=grid.ndim,
            num_components=num_components,
            lower_bound=grid.lower_bound,
            upper_bound=grid.upper_bound,
            num_gridpoints=grid.num_gridpoints,
            comm=grid.comm if distributed else None,
            num_ghost_points=1 if distributed else 0,
            domain_decomposition=decomposition.decompose if distributed else None,
            periodic=grid.periodic,
        )

    @classmethod
    def from_array(
        cls,
        data: Array,
        lower_bound: tuple,
        upper_bound: tuple,
        comm: MPI.Comm | None = None,
        num_ghost_points: int = 0,
        domain_decomposition: Any = None,
        periodic: bool | tuple[bool, ...] | None = None,
        dtype: type | None = None,
    ) -> Self:
        """Create a field from global scalar (``num_gridpoints``) or component-last data."""
        data = xp.asarray(data, dtype=dtype)
        if data.ndim == 0:
            raise ValueError("Field.from_array requires at least one spatial axis")

        ndim = len(lower_bound)
        if data.ndim == ndim:
            num_gridpoints = data.shape
            num_components = 1
            field_data = data[..., None]
        elif data.ndim == ndim + 1:
            num_gridpoints = data.shape[:-1]
            num_components = data.shape[-1]
            field_data = data
        else:
            raise ValueError(
                f"Expected data with {ndim} or {ndim + 1} dimensions, got {data.ndim}",
            )

        field = cls(
            ndim=ndim,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            num_gridpoints=num_gridpoints,
            num_components=num_components,
            comm=comm,
            num_ghost_points=num_ghost_points,
            domain_decomposition=domain_decomposition,
            periodic=periodic,
            dtype=field_data.dtype,
        )
        field.set_field_data(field_data)
        return field

    def _like(self, num_components: int | None = None, dtype: Any = None) -> Self:
        """A zero field of this type on the same grid and split."""
        return type(self)(
            ndim=self.ndim,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            num_gridpoints=self.num_gridpoints,
            num_components=(
                self.num_components if num_components is None else num_components
            ),
            comm=self.comm,
            num_ghost_points=self.num_ghost_points,
            domain_decomposition=list(self._decomposition_of(self).decompose),
            periodic=tuple(self._decomposition_of(self).periodic),
            dtype=self.dtype if dtype is None else dtype,
        )

    def copy(self, copy_data: bool = True) -> Self:
        """Return a field on the same grid, with a copy of the data (or zeros)."""
        result = self._like()
        if copy_data:
            result._field_data = self.field_data.copy()
        return result

    def astype(self, dtype: type, copy: bool = True) -> Self:
        """Return a field with data converted to ``dtype``."""
        result = self.copy(copy_data=False)
        result._dtype = xp.dtype(dtype)
        result._field_data = self.field_data.astype(dtype, copy=copy)
        return result

    @property
    def distributed(self) -> bool:
        """Whether the ranks hold different parts of the grid."""
        return bool(self.field_data.layout.distributed)

    # Setting and getting data ------------------------------------------------#

    @staticmethod
    def _fill(array: DistributedArray, data: Array | float) -> None:
        """Set every node from a global array or scalar and zero the halos."""
        if not xp.isscalar(data) and array.local_with_halos.shape == data.shape:
            # the whole array, without halo nodes, on this rank: a plain copy
            array.local_with_halos[...] = data
            return
        array[...] = data
        array.clear_halos()

    def set_field_data(
        self, data: Array | float | DistributedArray, component: int | None = None
    ) -> None:
        """Set the field from a scalar, a global array or a distributed array.

        - a scalar sets every stored node (of one ``component``, or all);
        - a global array of the field's shape ``num_gridpoints + (ncomp,)``, or of
          ``num_gridpoints`` for one ``component`` (or a scalar field): each rank
          copies its own nodes and the halos are zeroed;
        - a :class:`~mpiarray.DistributedArray` with the same split (the field's
          shape, or the grid's for one ``component``) is copied block by block,
          halo nodes included, without communication;
        - an array of the stored block's shape (:attr:`local_with_halos`) is copied as is.
        """
        if component is not None and not 0 <= component < self.num_components:
            raise IndexError(
                f"component {component} is out of range for a field with "
                f"{self.num_components} components"
            )
        if isinstance(data, DistributedArray):
            target = self.local_with_halos
            if component is not None:
                target = target[..., component]
            if data.local_with_halos.shape != target.shape:
                raise ValueError(
                    f"Expected a stored block of shape {target.shape}, "
                    f"got {data.local_with_halos.shape}"
                )
            target[...] = data.local_with_halos
            return
        if xp.isscalar(data):
            if component is None:
                self.local_with_halos[...] = data
            else:
                self.local_with_halos[..., component] = data
            return

        data = xp.asarray(data, dtype=self.dtype)

        if component is None and data.shape == self.field_data.shape:
            self._fill(self.field_data, data)
            return

        if (
            component is None
            and self.num_components == 1
            and data.shape == self.num_gridpoints
        ):
            component = 0

        if component is not None and data.shape == self.num_gridpoints:
            stored = self.local_with_halos
            if stored.shape[:-1] == data.shape:
                # the whole grid, without halo nodes, on this rank: a plain copy
                stored[..., component] = data
                return
            # each rank writes its own nodes; halos are zeroed as with ``_fill``
            self.field_data[..., component] = data
            self.field_data.clear_halos()
            return

        target = (
            self.local_with_halos
            if component is None
            else self.local_with_halos[..., component]
        )
        if target.shape != data.shape:
            raise ValueError(
                f"Expected data of shape {self.field_data.shape} (global), "
                f"{self.num_gridpoints} (one component) or {target.shape} (stored "
                f"block), got {data.shape}"
            )
        target[...] = data

    def get_field_data(
        self,
        coordinate: tuple[int, ...] | int | None = None,
        component: int | None = None,
        gathered: bool = False,
    ) -> Array:
        """Return the stored block (or the gathered field), optionally one node or component.

        ``coordinate`` indexes the stored block (halo nodes included), or the global
        field with ``gathered=True``; gathering is collective.
        """
        if isinstance(coordinate, int):
            coordinate = (coordinate,)
        source = self.field_data.gather() if gathered else self.local_with_halos
        if coordinate is not None:
            if len(coordinate) != self.ndim:
                raise ValueError(
                    f"coordinate needs {self.ndim} indices, got {len(coordinate)}"
                )
            data = source[coordinate]
        else:
            data = source
        if component is not None:
            return data[..., component]
        return data

    def _check_index(self, index: tuple, component: int) -> None:
        if len(index) != self.ndim:
            raise ValueError(f"index needs {self.ndim} entries, got {len(index)}")
        if not 0 <= component < self.num_components:
            raise IndexError(f"invalid component {component}")

    def set_value(self, index: IndexLike, value: float, component: int = 0) -> None:
        """Set the node at global ``index``; ranks that do not own it do nothing."""
        if isinstance(index, int):
            index = (index,)
        self._check_index(index, component)
        local_index = []
        for d in range(self.ndim):
            lower, upper = self.proc_index_bounds[d]
            if not lower <= index[d] < upper:
                return
            local_index.append(index[d] - lower + self.num_ghost_points)
        self.local_with_halos[tuple(local_index) + (component,)] = value

    def get_value(self, index: IndexLike, component: int = 0) -> Any:
        """Return the node at global ``index`` on every rank (an ``allreduce``: collective)."""
        if isinstance(index, int):
            index = (index,)
        self._check_index(index, component)
        owns = True
        local_index = []
        for d in range(self.ndim):
            low, high = self.proc_index_bounds[d]
            if not low <= index[d] < high:
                owns = False
                break
            local_index.append(index[d] - low + self.num_ghost_points)
        value = (
            self.local_with_halos[tuple(local_index) + (component,)] if owns else 0.0
        )
        if self.mpi_size > 1:
            # only the owner contributes a value
            value = self.comm.allreduce(value, op=MPI.SUM)
        return value

    def set_inside(self, shape: Shape, value: float, component: int = 0) -> None:
        """Set the stored nodes inside ``shape`` (a simgeometry shape) to ``value``.

        Every rank sets its own block, halo nodes included, from the shape's
        vectorised ``contains_points``; no communication.
        """
        if not 0 <= component < self.num_components:
            raise IndexError(f"invalid component {component}")
        grid = self.block_grid(cpu=True)
        lower = grid.bounds.lower_bound
        axes = [
            lower[d] + grid.grid_spacing[d] * np.arange(grid.num_gridpoints[d])
            for d in range(self.ndim)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack([m.ravel() for m in mesh])
        inside = np.asarray(xp.to_numpy(shape.contains_points(points)), dtype=bool)
        mask = xp.asarray(inside.reshape(tuple(grid.num_gridpoints)))
        self.local_with_halos[..., component][mask] = self.dtype.type(value)

    def to_ndarray(self) -> Array:
        """Return the global field data, ``num_gridpoints + (ncomp,)`` (collective)."""
        return self.field_data.gather()

    def to_numpy(self) -> np.ndarray:
        """Return the gathered field data as a ``numpy.ndarray`` (collective)."""
        return self.field_data.to_numpy()

    def to_cupy(self) -> Array:
        """Return the gathered field data on the active cunumpy backend (collective)."""
        return xp.asarray(self.field_data.gather())

    @property
    def local(self) -> Array:
        """This rank's nodes, without halo nodes: a writable view."""
        return self.field_data.local

    @property
    def local_with_halos(self) -> Array:
        """This rank's stored block, halo nodes included: a writable view.

        This is the array the kernels read and write, on the grid of
        :meth:`block_grid`.
        """
        return self.field_data.local_with_halos

    def component(self, component: int) -> DistributedArray:
        """Return one component as a distributed array sharing this field's storage."""
        if not 0 <= component < self.num_components:
            raise IndexError(f"invalid component {component}")
        return DistributedArray(
            self._layout(1, tuple(self.field_data.layout.periodic)),
            self.local_with_halos[..., component : component + 1],
        )

    # Halos -------------------------------------------------------------------#

    def fill_halos(self) -> None:
        """Fill the halo nodes from the neighbouring ranks (``update_halos``)."""
        self.field_data.update_halos()

    def exchange_halos(self) -> None:
        """Add the halo nodes to their owners on the neighbouring ranks and clear them.

        Call it after a deposit (:meth:`scatter`) into a distributed field.
        """
        self.field_data.accumulate_halos()

    # Reductions and filters --------------------------------------------------#

    @staticmethod
    def _global_reduction(result: Any) -> Any:
        """Return a reduction as a host scalar, or gathered as an array for ``axis``."""
        if isinstance(result, DistributedArray):
            return result.gather()
        return result

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        component: int | None = None,
    ) -> Any:
        """Sum the field over all ranks, optionally for one component."""
        data = self.field_data if component is None else self.component(component)
        return self._global_reduction(data.sum(axis=axis))

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        component: int | None = None,
    ) -> Any:
        """Return the mean over all ranks, optionally for one component."""
        data = self.field_data if component is None else self.component(component)
        return self._global_reduction(data.mean(axis=axis))

    def min(
        self,
        axis: int | tuple[int, ...] | None = None,
        component: int | None = None,
    ) -> Any:
        """Return the minimum over all ranks, optionally for one component."""
        data = self.field_data if component is None else self.component(component)
        return self._global_reduction(data.min(axis=axis))

    def max(
        self,
        axis: int | tuple[int, ...] | None = None,
        component: int | None = None,
    ) -> Any:
        """Return the maximum over all ranks, optionally for one component."""
        data = self.field_data if component is None else self.component(component)
        return self._global_reduction(data.max(axis=axis))

    @staticmethod
    def _resolve_smoothing_kernel(
        filter_name: Literal["binomial", "gaussian", "uniform"],
    ) -> Array:
        """Return a normalized 1-D kernel for separable smoothing."""
        kernels = {
            "binomial": xp.asarray([1.0, 2.0, 1.0], dtype=float),
            "gaussian": xp.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float),
            "uniform": xp.asarray([1.0, 1.0, 1.0], dtype=float),
        }
        if filter_name not in kernels:
            raise ValueError(
                f"Unsupported smoothing filter {filter_name!r}. "
                "Expected one of: 'binomial', 'gaussian', 'uniform'.",
            )
        kernel = kernels[filter_name]
        return kernel / xp.sum(kernel)

    @staticmethod
    def _convolve_axis(
        data: Array,
        kernel: Array,
        axis: int,
        periodic: bool,
    ) -> Array:
        """Apply a 1-D convolution along one axis, preserving the input shape."""
        radius = int(kernel.size // 2)
        if radius == 0:
            return data.copy()

        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (radius, radius)
        pad_mode = "wrap" if periodic else "constant"
        padded = xp.pad(data, pad_width=pad_width, mode=pad_mode)

        filtered = xp.zeros_like(data, dtype=float)
        for offset, weight in enumerate(kernel):
            slices = [slice(None)] * data.ndim
            slices[axis] = slice(offset, offset + data.shape[axis])
            filtered += weight * padded[tuple(slices)]
        return filtered

    @sp.profile("mpifield.Field.smooth", tags=PROFILE_TAGS)
    def smooth(
        self,
        filter_name: Literal["binomial", "gaussian", "uniform"] = "binomial",
        passes: int = 1,
        component: int | None = None,
        in_place: bool = True,
    ) -> Self:
        """Smooth the field with a separable low-pass filter.

        Smoothing is applied independently per component and per axis. For
        periodic axes, values wrap around. For non-periodic axes, out-of-domain
        values are treated as zero. Under MPI it exchanges halos of the filter's
        width when every rank's block is at least that wide, and gathers otherwise.
        """
        if passes < 1:
            raise ValueError("passes must be >= 1")
        if component is not None and not (0 <= component < self.num_components):
            raise ValueError(
                f"component must be in [0, {self.num_components - 1}], got {component}",
            )

        target = self if in_place else self.copy()
        kernel = self._resolve_smoothing_kernel(filter_name=filter_name)
        periodic = tuple(
            bool(flag) for flag in target.field_data.layout.periodic[: self.ndim]
        )
        comp_indices = (
            [component] if component is not None else list(range(target.num_components))
        )

        radius = int(kernel.size // 2)
        if target._can_smooth_with_halos(radius):
            target._smooth_with_halos(kernel, passes, comp_indices)
            return target

        gathered = target.to_ndarray()
        for comp in comp_indices:
            smoothed = xp.asarray(gathered[..., comp], dtype=float)
            for _ in range(passes):
                for axis in range(target.ndim):
                    smoothed = target._convolve_axis(
                        smoothed,
                        kernel=kernel,
                        axis=axis,
                        periodic=periodic[axis],
                    )
            gathered[..., comp] = smoothed

        self._fill(target.field_data, gathered.astype(target.dtype, copy=False))
        return target

    def _can_smooth_with_halos(self, radius: int) -> bool:
        """Return whether smoothing can run on each rank's block plus halos.

        That needs a distributed field and every rank's block at least
        ``radius`` nodes wide, so one neighbour supplies each halo. All ranks
        reach the same answer without communicating.
        """
        layout = self.field_data.layout
        if radius == 0 or not layout.distributed:
            return False
        return all(
            min(layout.local_shape_of(rank)[axis] for rank in range(layout.size))
            >= radius
            for axis in range(self.ndim)
        )

    def _smooth_with_halos(
        self,
        kernel: Array,
        passes: int,
        comp_indices: list[int],
    ) -> None:
        """Smooth in place, exchanging only halos of width ``kernel.size // 2``.

        Gives the same result as smoothing the gathered array: filled halos
        wrap on periodic axes and stay zero at physical boundaries.
        """
        field_data = self.field_data
        radius = int(kernel.size // 2)
        work = mpa.zeros(
            dtype=float,
            layout=self._layout(
                self.num_components,
                tuple(field_data.layout.periodic),
                halo=radius,
            ),
        )
        work.local[...] = field_data.local
        interior = work.layout.interior
        for _ in range(passes):
            for axis in range(self.ndim):
                work.update_halos(axis)
                # the full extent (with halos) along ``axis``, the interior elsewhere
                padded_index = list(interior)
                padded_index[axis] = slice(None)
                padded = work.local_with_halos[tuple(padded_index)]
                num_cells = padded.shape[axis] - 2 * radius
                filtered = xp.zeros_like(work.local)
                for offset, weight in enumerate(kernel):
                    window = [slice(None)] * padded.ndim
                    window[axis] = slice(offset, offset + num_cells)
                    filtered += weight * padded[tuple(window)]
                work.local[...] = filtered

        for comp in comp_indices:
            field_data.local[..., comp] = work.local[..., comp]
        field_data.clear_halos()

    def smooth_binomial(
        self, passes: int = 1, component: int | None = None, in_place: bool = True
    ) -> Self:
        """Smooth the field with the 3-point binomial kernel ``[1, 2, 1] / 4``."""
        return self.smooth("binomial", passes, component, in_place)

    def smooth_gaussian(
        self, passes: int = 1, component: int | None = None, in_place: bool = True
    ) -> Self:
        """Smooth the field with the 5-point Gaussian-like kernel ``[1, 4, 6, 4, 1] / 16``."""
        return self.smooth("gaussian", passes, component, in_place)

    def smooth_uniform(
        self, passes: int = 1, component: int | None = None, in_place: bool = True
    ) -> Self:
        """Smooth the field with a 3-point moving-average kernel."""
        return self.smooth("uniform", passes, component, in_place)

    # Derivatives -------------------------------------------------------------#

    @sp.profile("mpifield.Field.gradient", tags=PROFILE_TAGS)
    def gradient(self) -> Self:
        """Return the gradient of a scalar field: a field with ``ndim`` components.

        Second-order central differences inside the domain and across a periodic
        seam, first-order one-sided differences at the ends of a bounded axis (as
        :func:`numpy.gradient`). A distributed field is gathered first (collective),
        so the result does not depend on the split.
        """
        if self.num_components != 1:
            raise ValueError(
                f"gradient() needs a scalar field, this one has {self.num_components} "
                "components"
            )
        data = xp.asarray(
            self.to_ndarray()[..., 0] if self.distributed else self.local[..., 0],
            dtype=float,
        )
        periodic = _periodic_flags(self.periodic, self.ndim)
        components = []
        for axis in range(self.ndim):
            spacing = float(self.grid_spacing[axis])
            if periodic[axis]:
                derivative = (
                    xp.roll(data, -1, axis=axis) - xp.roll(data, 1, axis=axis)
                ) / (2.0 * spacing)
            else:
                derivative = xp.gradient(data, spacing, axis=axis)
            components.append(derivative)
        result = self._like(num_components=self.ndim, dtype=float)
        stacked = xp.stack(components, axis=-1)
        if self.distributed:
            result.set_field_data(stacked)
        else:
            result.local[...] = stacked
        return result

    # Evaluation and plotting -------------------------------------------------#

    def evaluate(
        self,
        x: Any = None,
        y: Any = None,
        z: Any = None,
        component: int | None = None,
        name: str | None = None,
        units: str | None = None,
    ) -> xr.DataArray:
        """Evaluate the field as an ``xarray.DataArray`` with ``x``, ``y``, ``z`` dims.

        Each axis takes a scalar, a list, a one-dimensional array or a ``range``
        in physical coordinates. A scalar fixes that axis and drops its dimension
        (it stays as a scalar coordinate); a sequence becomes a dimension, and
        several sequences form their tensor-product mesh. An omitted axis keeps
        the grid nodes, so ``field.evaluate()`` is the field data itself, without
        interpolation. For scattered points, pass ``xarray.DataArray`` coordinates
        that share a dimension, e.g. ``dims="points"``. Values between nodes are
        interpolated linearly; periodic axes wrap around. Coordinates outside the
        domain raise ``ValueError``.

        A field with several components has a ``component`` dimension, unless one
        is chosen with ``component=``; a single-component field has none.
        ``name`` and ``units`` label the result (``units`` goes to
        ``attrs["units"]``, which plots show on the axis or colorbar).

        Under MPI this gathers the global field, so call it on every rank.

        Examples:
        --------
        >>> phi.evaluate()  # (x, y) on the grid nodes
        >>> phi.evaluate(y=0.0)  # a line profile along x
        >>> phi.evaluate(x=np.linspace(0, 1, 200), y=[0.25, 0.5])
        >>> E.evaluate(z=0.1, component=0)
        """
        from mpifield.evaluate import evaluate_field

        return evaluate_field(
            self,
            {"x": x, "y": y, "z": z},
            component=component,
            name=name,
            units=units,
        )

    def plot(
        self,
        ax: Any = None,
        x: Any = None,
        y: Any = None,
        z: Any = None,
        component: int | None = None,
        name: str | None = None,
        units: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the field with plasma-plots and return its ``PlotResult``.

        ``x``, ``y``, ``z``, ``component``, ``name`` and ``units`` are as in
        :meth:`evaluate`. A one-dimensional result is drawn as a line, a
        two-dimensional one as a colour map. A 3D field with no axis fixed is
        shown on its ``z`` midplane. Other keyword arguments (``cmap``, ``vmin``,
        ``title``, ``backend``, ...) go to ``plasma_plots``. Needs the ``plot``
        extra.

        Under MPI, call it on every rank: the field is gathered everywhere.

        Examples:
        --------
        >>> phi.plot()
        >>> phi.plot(y=0.5)  # line profile
        >>> E.plot(ax=ax, component=1, cmap="RdBu_r", symmetric=True)
        """
        from mpifield.evaluate import plot_field

        return plot_field(
            self,
            {"x": x, "y": y, "z": z},
            component=component,
            name=name,
            units=units,
            ax=ax,
            **kwargs,
        )

    # Gather and scatter ------------------------------------------------------#

    def _marker_positions(self, positions: Array | float) -> Array:
        """Return ``positions`` as a ``(ndim, N)`` float array.

        A float (1-D fields) or a 1-D array of length ``ndim`` is one point; a
        1-D array is otherwise ``N`` points of a 1-D field.
        """
        if isinstance(positions, float):
            if self.ndim != 1:
                raise ValueError(
                    f"a float is one point of a 1D field, not {self.ndim}D"
                )
            return xp.array([[positions]], dtype=float)

        if isinstance(positions, (list, tuple, np.ndarray)) or xp.is_gpu(positions):
            positions = xp.asarray(positions, dtype=float)
            if positions.ndim == 1:
                if self.ndim == 1:
                    return positions.reshape((1, -1))  # (1, N)
                if positions.shape[0] != self.ndim:
                    raise ValueError(
                        f"Expected {self.ndim} coordinates, got {positions.shape[0]}"
                    )
                return positions.reshape((self.ndim, 1))  # one point
            if positions.ndim == 2:
                if positions.shape[0] != self.ndim:
                    raise ValueError(
                        f"Expected positions with shape ({self.ndim}, N), "
                        f"got {positions.shape}"
                    )
                return positions
            raise ValueError(f"Unsupported shape: {positions.shape}")

        raise TypeError(
            "Unsupported format for positions: "
            f"type={type(positions)}, "
            f"shape={getattr(positions, 'shape', 'N/A')}",
        )

    def _marker_values(self, values: Array | float, num_positions: int) -> Array:
        """Return ``values`` as a ``(num_components, N)`` float array.

        A number is the same value for every marker and component; a 1-D array is
        ``N`` values of a scalar field, or the components of one marker.
        """
        if isinstance(values, (float, int)):
            return xp.full((self.num_components, num_positions), values, dtype=float)
        if not (isinstance(values, (list, tuple, np.ndarray)) or xp.is_gpu(values)):
            raise TypeError(
                "Unsupported format for values: "
                f"type={type(values)}, "
                f"shape={getattr(values, 'shape', 'N/A')}",
            )
        values = xp.asarray(values, dtype=float)
        if values.ndim == 1:
            if values.shape[0] == num_positions:
                values = values.reshape((1, num_positions))
            elif num_positions == 1 and values.shape[0] == self.num_components:
                values = values.reshape((self.num_components, 1))
            else:
                raise ValueError(
                    f"Expected {num_positions} values (or {self.num_components} "
                    f"components of one marker), got {values.shape}"
                )
        elif values.ndim != 2 or values.shape[1] != num_positions:
            raise ValueError(
                f"Expected values with shape ({self.num_components}, {num_positions}), "
                f"got {values.shape}"
            )
        if values.shape[0] != self.num_components:
            raise ValueError(
                f"Expected {self.num_components} components, got {values.shape[0]}"
            )
        return values

    def _check_halo(self, weighting: Weighting) -> None:
        """Raise if this field is split with fewer halo nodes than ``weighting`` needs."""
        if self.num_ghost_points < weighting.halo and self.distributed:
            raise ValueError(
                f"the {weighting.name} weighting needs num_ghost_points >= "
                f"{weighting.halo} on a field split over ranks, this one has "
                f"{self.num_ghost_points}"
            )

    def _host_grid(self) -> BlockGrid:
        """:meth:`block_grid` on the CPU inside the domain, cached for the current layout.

        Checked by identity, without hashing the layout: the deposit of every step
        takes this path.
        """
        layout = self.field_data.layout
        cached = self.__dict__.get("_host_block_grid")
        if cached is None or cached[0] is not layout:
            cached = (layout, self.block_grid(cpu=True, inside_domain=True))
            self.__dict__["_host_block_grid"] = cached
        return cached[1]

    @sp.profile("mpifield.Field.gather", tags=PROFILE_TAGS)
    def gather(self, positions: Array | float, weighting: Weighting = LINEAR) -> Array:
        """Interpolate the field at marker positions with ``weighting``.

        ``positions`` has shape ``(ndim, N)`` (or is one point as a 1-D array of
        length ``ndim``); the result has shape ``(num_components, N)``. With the
        built-in weightings a position outside the grid uses its edge nodes; on a
        periodic axis without halos it wraps around. Under MPI each rank
        interpolates from its own block, so its markers must lie in it, and the
        halo nodes must be filled (:meth:`fill_halos`) for markers near its edge.
        """
        self._check_halo(weighting)
        positions = self._marker_positions(positions)
        stored = self.local_with_halos

        if xp.is_gpu(positions) and xp.is_gpu(stored):
            # gather on the GPU: no copies of the markers or the field
            import cupy  # pyright: ignore[reportMissingImports]  # the GPU extra

            grid = self.block_grid(inside_domain=True)
            result = cupy.zeros((self.num_components, positions.shape[1]))
            weighting.gather_block(positions, stored[grid.index], result, grid)
            return result

        grid = self._host_grid()
        result_cpu = np.zeros((self.num_components, positions.shape[1]), dtype=float)
        weighting.gather_block(
            xp.to_numpy(positions), xp.to_numpy(stored)[grid.index], result_cpu, grid
        )
        return xp.asarray(result_cpu)

    @sp.profile("mpifield.Field.scatter", tags=PROFILE_TAGS)
    def scatter(
        self,
        positions: Array | float,
        values: Array | float,
        weighting: Weighting = LINEAR,
    ) -> None:
        """Add marker values to the nodes with ``weighting`` (linear by default).

        ``positions`` has shape ``(ndim, N)`` and ``values`` ``(ncomp, N)``; a 1-D
        ``values`` of length ``N`` is one component, a number is the same value
        for every marker. The values are added to the stored block, halo nodes
        included; under MPI call :meth:`exchange_halos` afterwards. A marker with
        a zero value contributes nothing, whatever its position.
        """
        self._check_halo(weighting)
        data = self.local_with_halos
        if (
            type(positions) is np.ndarray
            and type(values) is np.ndarray
            and type(data) is np.ndarray
            and positions.ndim == 2
            and values.ndim == 2
            and positions.shape[0] == self.ndim
            and values.shape[0] == self.num_components
            and values.shape[1] == positions.shape[1]
            and positions.dtype == np.float64
            and values.dtype == np.float64
        ):
            # the deposit of every step: host marker arrays as the compiled
            # kernels take them, with nothing to check or convert
            grid = self._host_grid()
            weighting.scatter_block(positions, values, data[grid.index], grid)
            return
        positions = self._marker_positions(positions)
        values = self._marker_values(values, positions.shape[1])

        if xp.is_gpu(positions) and xp.is_gpu(data):
            # deposit on the GPU: no copies of the markers or the field
            grid = self.block_grid(inside_domain=True)
            weighting.scatter_block(positions, values, data[grid.index], grid)
            return

        grid = self._host_grid()
        field_data_cpu = xp.to_numpy(data)
        weighting.scatter_block(
            xp.to_numpy(positions),
            xp.to_numpy(values),
            field_data_cpu[grid.index],
            grid,
        )
        if field_data_cpu is not data:
            data[...] = xp.asarray(field_data_cpu)

    # Arithmetic --------------------------------------------------------------#

    def _aligned_data(self, other: Field) -> DistributedArray:
        """Return ``other``'s data with this field's layout.

        Layouts that differ only in periodicity, which matters for halo exchanges
        but not for arithmetic, share storage with ``other``; any other
        difference raises.
        """
        mine = self.field_data.layout
        theirs = other.field_data.layout
        if mine == theirs:
            return other.field_data
        if (
            mine.shape == theirs.shape
            and mine.halo == theirs.halo
            and mine.process_grid == theirs.process_grid
            and mine.bounds == theirs.bounds
            and mine.comm == theirs.comm
        ):
            return DistributedArray(mine, other.field_data.local_with_halos)
        self.field_data._check_compatibility(other.field_data)
        return other.field_data

    def _check_field_compatibility(self, other: Field, components: bool = True) -> None:
        """Raise unless ``other`` is a field on the same grid and split.

        With ``components=False`` the component counts may differ.
        """
        if not isinstance(other, Field):
            raise TypeError(f"expected a Field, got {type(other).__name__}")
        if self.ndim != other.ndim:
            raise ValueError("Fields must have the same number of dimensions")
        if components and self.num_components != other.num_components:
            raise ValueError("Number of components must match")
        if not xp.allclose(self.lower_bound, other.lower_bound):
            raise ValueError("lower_bound must be the same")
        if not xp.allclose(self.upper_bound, other.upper_bound):
            raise ValueError("upper_bound must be the same")
        if self.num_gridpoints != other.num_gridpoints:
            raise ValueError("Grid resolution must match")
        if components:
            self._aligned_data(other)

    def _stretched(self, num_components: int, like: Field) -> DistributedArray:
        """This scalar field's data broadcast to ``num_components``, in ``like``'s layout.

        A read-only view, for the operand of an elementwise operation.
        """
        like._check_field_compatibility(self, components=False)
        shape = like.local_with_halos.shape
        if shape[:-1] != self.local_with_halos.shape[:-1]:
            raise ValueError("Fields must have the same split over the ranks")
        return DistributedArray(
            like.field_data.layout, xp.broadcast_to(self.local_with_halos, shape)
        )

    def _coerce_operand(self, other: Any) -> Any:
        """Return a value suitable for DistributedArray field operations."""
        if isinstance(other, Field):
            return self._aligned_data(other)
        if xp.isscalar(other):
            return other

        other = xp.asarray(other)
        if other.shape == self.num_gridpoints and self.num_components == 1:
            expanded = xp.zeros(self.field_data.shape, dtype=other.dtype)
            expanded[..., 0] = other
            return expanded
        return other

    def _operands(self, other: Any) -> tuple[Field, Any, Any]:
        """Return ``(template, own data, other operand)`` for an elementwise operation.

        A scalar field (one component) broadcasts over the components of a vector
        field, on either side; the template is then the vector field, whose type,
        grid and components the result takes.
        """
        if not isinstance(other, Field) or other.num_components == self.num_components:
            return self, self.field_data, self._coerce_operand(other)
        if other.num_components == 1:
            return self, self.field_data, other._stretched(self.num_components, self)
        if self.num_components == 1:
            return other, self._stretched(other.num_components, other), other.field_data
        raise ValueError(
            f"cannot combine fields with {self.num_components} and "
            f"{other.num_components} components; one of them may have one"
        )

    def _binary(self, other: Any, op: Any, reflected: bool = False) -> Self:
        """Apply an elementwise binary operation; ``reflected`` swaps the operands."""
        template, mine, theirs = self._operands(other)
        data = op(theirs, mine) if reflected else op(mine, theirs)
        return template._new_with_data(data)

    def _new_with_data(self, field_data: DistributedArray) -> Self:
        """Create a matching field that owns the provided distributed data."""
        result = self.copy(copy_data=False)
        result._field_data = field_data
        result._dtype = field_data.dtype
        result._num_components = int(field_data.shape[-1])
        return result

    def _binary_iop(self, other: Any, op: Any) -> Self:
        """Apply an in-place elementwise operation to field data."""
        if isinstance(other, Field) and other.num_components != self.num_components:
            if other.num_components != 1:
                raise ValueError(
                    f"cannot update a field with {self.num_components} components "
                    f"by one with {other.num_components}"
                )
            other_data = other._stretched(self.num_components, self).local_with_halos
        else:
            other_data = self.field_data._coerce_other_data(self._coerce_operand(other))
        op(self.local_with_halos, other_data, out=self.local_with_halos)
        return self

    def cross(self, other: Self) -> Self:
        """Return the pointwise cross product of two 3-component fields."""
        if self.num_components != 3 or other.num_components != 3:
            raise ValueError("the cross product needs fields with 3 components")
        self._check_field_compatibility(other)
        return self._new_with_data(
            DistributedArray(
                self.field_data.layout,
                xp.cross(self.local_with_halos, other.local_with_halos),
            ),
        )

    def dot(self, other: Self) -> Self:
        """Return the pointwise dot product as a scalar field."""
        self._check_field_compatibility(other)
        result = self._like(
            num_components=1, dtype=xp.result_type(self.dtype, other.dtype)
        )
        result.local_with_halos[..., 0] = xp.sum(
            self.local_with_halos * other.local_with_halos, axis=-1
        )
        return result

    def magnitude(self) -> Self:
        """Return the pointwise vector magnitude as a scalar field."""
        result = self._like(num_components=1, dtype=float)
        result.local_with_halos[..., 0] = xp.sqrt(
            xp.sum(self.local_with_halos * self.local_with_halos, axis=-1)
        )
        return result

    # I/O ---------------------------------------------------------------------#

    def export(self, file_path: PathLikeStr) -> None:
        """Write the field to a NetCDF file (``.nc``), through xarray.

        The file holds the node values (``x``, ``y``, ``z`` and, for several
        components, ``component``), the bounds and the periodic flags; the split
        over ranks is not stored. Collective under MPI: rank 0 writes.
        """
        array = self.evaluate(name="field")
        array.attrs.update(
            lower_bound=[float(v) for v in self.lower_bound],
            upper_bound=[float(v) for v in self.upper_bound],
            periodic=[int(flag) for flag in _periodic_flags(self.periodic, self.ndim)],
            num_components=self.num_components,
        )
        if self.mpi_rank == 0:
            array.to_netcdf(file_path)
        if self.mpi_size > 1:
            self.comm.Barrier()

    @classmethod
    def load(
        cls,
        file_path: PathLikeStr,
        comm: MPI.Comm | None = None,
        num_ghost_points: int = 0,
        domain_decomposition: Any = None,
    ) -> Self:
        """Read a field written by :meth:`export`, split as the arguments say."""
        import xarray as xr

        with xr.open_dataarray(file_path) as array:
            array = array.load()
        data = np.asarray(array.values)
        num_components = int(array.attrs["num_components"])
        if num_components == 1 and "component" in array.dims:
            data = data[..., 0]
        periodic = tuple(bool(flag) for flag in np.atleast_1d(array.attrs["periodic"]))
        return cls.from_array(
            data,
            lower_bound=tuple(
                float(v) for v in np.atleast_1d(array.attrs["lower_bound"])
            ),
            upper_bound=tuple(
                float(v) for v in np.atleast_1d(array.attrs["upper_bound"])
            ),
            comm=comm,
            num_ghost_points=num_ghost_points,
            domain_decomposition=domain_decomposition,
            periodic=periodic if any(periodic) else None,
        )

    # Properties --------------------------------------------------------------#

    @property
    def dtype(self) -> Any:
        """Return the data type."""
        return self._dtype

    @property
    def num_components(self) -> int:
        """Return the number of field components."""
        return self._num_components

    @property
    def field_data(self) -> DistributedArray:
        """Return the distributed array holding the field."""
        return self._field_data

    @property
    def num_ghost_points(self) -> int:
        """Return the number of halo nodes on each side of a rank's block."""
        return self._num_ghost_points

    # Python protocols --------------------------------------------------------#

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Array:
        """Return the gathered field data for NumPy interoperability."""
        array = self.to_ndarray()
        if dtype is not None:
            return array.astype(dtype, copy=False if copy is None else copy)
        if copy:
            return array.copy()
        return array

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any):
        """Apply NumPy/CuPy ufuncs elementwise to field data."""
        if method != "__call__":
            return NotImplemented
        if kwargs.get("out") is not None:
            return NotImplemented

        template = next((item for item in inputs if isinstance(item, Field)), self)
        darray_inputs = []
        for item in inputs:
            if isinstance(item, Field):
                template._check_field_compatibility(item)
                darray_inputs.append(template._aligned_data(item))
            else:
                darray_inputs.append(template._coerce_operand(item))

        result = getattr(ufunc, method)(*darray_inputs, **kwargs)
        if isinstance(result, tuple):
            return tuple(template._new_with_data(item) for item in result)
        return template._new_with_data(result)

    def __getitem__(self, index: IndexLike) -> Any:
        """Return global node values (``field_data[index]``)."""
        return self.field_data[index]

    def __setitem__(self, index: IndexLike, value: Any) -> None:
        """Set global node values (``field_data[index] = value``)."""
        self.field_data[index] = value

    def __add__(self, other: Any) -> Self:
        """Return the element-wise sum with a scalar, an array or a compatible field."""
        return self._binary(other, operator.add)

    def __radd__(self, other: Any) -> Self:
        """Return the reflected element-wise sum."""
        return self._binary(other, operator.add, reflected=True)

    def __iadd__(self, other: Any) -> Self:
        """Add to this field in place."""
        return self._binary_iop(other, xp.add)

    def __sub__(self, other: Any) -> Self:
        """Return the element-wise difference."""
        return self._binary(other, operator.sub)

    def __rsub__(self, other: Any) -> Self:
        """Return the reflected element-wise difference."""
        return self._binary(other, operator.sub, reflected=True)

    def __isub__(self, other: Any) -> Self:
        """Subtract from this field in place."""
        return self._binary_iop(other, xp.subtract)

    def __mul__(self, other: Any) -> Self:
        """Return the element-wise product."""
        return self._binary(other, operator.mul)

    def __rmul__(self, other: Any) -> Self:
        """Return the reflected element-wise product."""
        return self._binary(other, operator.mul, reflected=True)

    def __imul__(self, other: Any) -> Self:
        """Multiply this field in place."""
        return self._binary_iop(other, xp.multiply)

    def __truediv__(self, other: Any) -> Self:
        """Return the element-wise quotient."""
        return self._binary(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> Self:
        """Return the reflected element-wise quotient."""
        return self._binary(other, operator.truediv, reflected=True)

    def __itruediv__(self, other: Any) -> Self:
        """Divide this field in place."""
        return self._binary_iop(other, xp.divide)

    def __pow__(self, other: Any) -> Self:
        """Raise this field elementwise."""
        return self._binary(other, operator.pow)

    def __rpow__(self, other: Any) -> Self:
        """Raise values elementwise with reflected operands."""
        return self._binary(other, operator.pow, reflected=True)

    def __neg__(self) -> Self:
        """Return a field with all components negated."""
        return self._new_with_data(-self.field_data)

    def __abs__(self) -> Self:
        """Return absolute field values."""
        return self._new_with_data(abs(self.field_data))

    def __lt__(self, other: Any) -> Self:
        """Compare field data elementwise."""
        return self._binary(other, operator.lt)

    def __le__(self, other: Any) -> Self:
        """Compare field data elementwise."""
        return self._binary(other, operator.le)

    def __eq__(self, other: object) -> Self:  # type: ignore[override]
        """Compare field data elementwise."""
        return self._binary(other, operator.eq)

    def __ne__(self, other: object) -> Self:  # type: ignore[override]
        """Compare field data elementwise."""
        return self._binary(other, operator.ne)

    def __gt__(self, other: Any) -> Self:
        """Compare field data elementwise."""
        return self._binary(other, operator.gt)

    def __ge__(self, other: Any) -> Self:
        """Compare field data elementwise."""
        return self._binary(other, operator.ge)

    __hash__ = None  # type: ignore[assignment]  # mutable, compared elementwise

    def __repr__(self) -> str:
        """Return the developer-facing string representation."""
        return (
            f"{type(self).__name__}(ndim={self.ndim}, "
            f"num_gridpoints={self.num_gridpoints}, "
            f"num_components={self.num_components}, "
            f"lower_bound={tuple(float(v) for v in self.lower_bound)}, "
            f"upper_bound={tuple(float(v) for v in self.upper_bound)}, "
            f"distributed={self.distributed})"
        )

    def __str__(self) -> str:
        """Return a summary of the field and this rank's block."""
        return (
            f"{self!r}\n"
            f"rank {self.mpi_rank}: stored block {self.local_with_halos.shape}, "
            f"index bounds {self.proc_index_bounds}"
        )
