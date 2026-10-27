"""Weightings: how marker values are deposited onto the nodes and gathered back.

A :class:`Weighting` pairs a scatter (deposit) kernel and a gather (interpolation)
kernel for each number of dimensions it covers, and says how many halo nodes a
split field needs for it. :meth:`Field.scatter <mpifield.Field.scatter>` and
:meth:`Field.gather <mpifield.Field.gather>` take one; built in are

- :data:`LINEAR`: cloud-in-cell, two nodes per axis (the default);
- :data:`QUADRATIC`: the quadratic B-spline ("triangular-shaped cloud"), three
  nodes per axis, the nearest node and its neighbours.

The gather of a weighting is the transpose of its scatter, so that
``sum_i gather(G)(x_i) q_i == sum_n G_n scatter(q)_n``: a deposit and a gather
with different weightings break momentum and energy conservation, so they come
together.

The kernel convention: a weighting's kernels for ``ndim`` axes are called as::

    scatter(positions, values, field, *geometry)   # adds the values into field
    gather(positions, field, result, *geometry)    # overwrites result

with ``geometry = (*lower, *cell_size, *num_nodes, *periodic)``
(:attr:`BlockGrid.geometry <mpifield.BlockGrid>`: ``ndim`` floats, ``ndim``
floats, ``ndim`` ints, ``ndim`` ints 0 or 1). Node ``k`` of axis ``d`` is at
``lower[d] + k * cell_size[d]``; on a periodic axis the last node neighbours the
first. ``positions`` is ``(ndim, N)``, ``values`` and ``result`` ``(ncomp, N)``,
``field`` ``num_nodes + (ncomp,)``; all float64 and all NumPy arrays or all CuPy
arrays. ``field`` is C-contiguous; the marker arrays are C-ordered but may have
gaps between rows (``storage[:, :n]``). A marker whose values are all zero (a dead
one left in storage) should deposit nothing and not read its position, which may
be non-finite. A :class:`cunumpy.kernels.Kernel` made with ``from_folder`` fits:
it runs its CUDA version on CuPy arrays and its Pyccel or NumPy version on NumPy
ones. So does a plain Python function.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any

import cunumpy as xp

from mpifield._types import Array, BlockGrid
from mpifield.kernels.gather_linear_1d import kernel as gather_linear_1d
from mpifield.kernels.gather_linear_2d import kernel as gather_linear_2d
from mpifield.kernels.gather_linear_3d import kernel as gather_linear_3d
from mpifield.kernels.gather_quadratic_1d import kernel as gather_quadratic_1d
from mpifield.kernels.gather_quadratic_2d import kernel as gather_quadratic_2d
from mpifield.kernels.gather_quadratic_3d import kernel as gather_quadratic_3d
from mpifield.kernels.scatter_linear_1d import kernel as scatter_linear_1d
from mpifield.kernels.scatter_linear_1d.scatter_linear_1d_numpy import (
    scatter_linear_1d as scatter_linear_1d_numpy,
)
from mpifield.kernels.scatter_linear_2d import kernel as scatter_linear_2d
from mpifield.kernels.scatter_linear_3d import kernel as scatter_linear_3d
from mpifield.kernels.scatter_quadratic_1d import kernel as scatter_quadratic_1d
from mpifield.kernels.scatter_quadratic_2d import kernel as scatter_quadratic_2d
from mpifield.kernels.scatter_quadratic_3d import kernel as scatter_quadratic_3d

__all__ = ["LINEAR", "QUADRATIC", "Weighting"]

KernelFunction = Callable[..., Any]


def _frozen(kernels: Mapping[int, KernelFunction], role: str) -> Mapping[int, Any]:
    """Return ``kernels`` as a read-only mapping, checking the dimensions."""
    for ndim in kernels:
        if ndim not in (1, 2, 3):
            raise ValueError(f"{role} kernels are for 1, 2 or 3 dimensions, got {ndim}")
    return MappingProxyType(dict(kernels))


@dataclass(frozen=True, eq=False, repr=False)
class Weighting:
    """A deposit and its gather, for each number of dimensions they cover.

    ``scatter`` and ``gather`` map a number of dimensions (1, 2, 3) to a kernel
    with the arguments of the module's kernel convention; a weighting may cover
    some dimensions only. ``halo`` is the number of halo nodes a field split over
    ranks needs, so that every marker in a block's own cells deposits into and
    gathers from the stored block: for a stencil of ``s`` nodes per axis,
    ``(s + 1) // 2``.

    Examples:
    --------
    >>> def scatter_2d(positions, values, field, *geometry): ...
    >>> def gather_2d(positions, field, result, *geometry): ...
    >>> mine = Weighting("mine", scatter={2: scatter_2d}, gather={2: gather_2d}, halo=2)
    """

    name: str
    scatter: Mapping[int, KernelFunction]
    gather: Mapping[int, KernelFunction]
    halo: int = 1

    def __post_init__(self) -> None:
        """Check the dimensions and the halo; freeze the kernel mappings."""
        if self.halo < 0:
            raise ValueError(f"halo must be >= 0, got {self.halo}")
        object.__setattr__(self, "scatter", _frozen(self.scatter, "scatter"))
        object.__setattr__(self, "gather", _frozen(self.gather, "gather"))

    def _kernel(self, kernels: Mapping[int, KernelFunction], ndim: int, role: str):
        """Return the kernel for ``ndim`` dimensions, or raise naming the weighting."""
        try:
            return kernels[ndim]
        except KeyError:
            have = ", ".join(f"{n}D" for n in sorted(kernels)) or "none"
            raise ValueError(
                f"the {self.name} weighting has no {ndim}D {role} kernel (has {have})"
            ) from None

    def scatter_kernel(self, ndim: int) -> KernelFunction:
        """Return the scatter kernel for ``ndim`` dimensions."""
        return self._kernel(self.scatter, ndim, "scatter")

    def gather_kernel(self, ndim: int) -> KernelFunction:
        """Return the gather kernel for ``ndim`` dimensions."""
        return self._kernel(self.gather, ndim, "gather")

    def scatter_block(
        self, positions: Array, values: Array, field: Array, grid: BlockGrid
    ) -> None:
        """Deposit ``values`` at ``positions`` into the nodes of ``field``, a block on ``grid``.

        ``field`` is ``num_nodes + (ncomp,)``, on the CPU or the GPU; the markers
        are moved to its side, and a copy is made only where an array is not
        float64 or (for ``field``) not C-contiguous. A 1-D ``values`` is one
        component.
        """
        kernel = self.scatter_kernel(len(grid.periodic))
        convert = partial(
            xp.kernels.as_kernel_array, like=field, dtype=float, strided=True
        )
        positions = convert(positions)
        values = convert(values).reshape(-1, positions.shape[-1])
        with xp.kernels.kernel_output(field, like=field, dtype=float) as buffer:
            kernel(positions, values, buffer, *grid.geometry)

    def gather_block(
        self, positions: Array, field: Array, result: Array, grid: BlockGrid
    ) -> None:
        """Interpolate ``field``, a block on ``grid``, at ``positions`` into ``result``.

        ``result`` is ``(ncomp, N)`` and may be on either side of ``field``; the
        kernel runs on the side of ``field``.
        """
        kernel = self.gather_kernel(len(grid.periodic))
        positions = xp.kernels.as_kernel_array(
            positions, like=field, dtype=float, strided=True
        )
        grid_values = xp.kernels.as_kernel_array(field, like=field, dtype=float)
        with xp.kernels.kernel_output(
            result, like=field, dtype=float, strided=True
        ) as out:
            kernel(positions, grid_values, out, *grid.geometry)

    def __repr__(self) -> str:
        """Name the weighting, its dimensions and its halo."""
        dims = sorted(set(self.scatter) | set(self.gather))
        return f"Weighting({self.name!r}, ndim={tuple(dims)}, halo={self.halo})"


def _scatter_linear_1d(
    positions: Array,
    values: Array,
    field: Array,
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
) -> None:
    """LINEAR's 1D deposit: the compiled kernel, one component at a time.

    On the GPU each block deposits into a copy of the grid in shared memory when
    it fits, reusing it for about four markers per thread (fewer grid
    initializations and global atomics); otherwise the NumPy version runs on the
    CuPy arrays.
    """
    args = (lower_x, cell_size_x, num_x, periodic_x)
    gpu = xp.is_gpu(field)
    if gpu and not xp.cuda.shared_memory_fits(num_x):
        for component in range(field.shape[1]):
            scatter_linear_1d_numpy(
                positions[0], values[component], field[:, component], *args
            )
        return
    launch = {"n_threads": (positions.shape[1] + 3) // 4} if gpu else {}
    for component in range(field.shape[1]):
        with xp.kernels.kernel_output(
            field[:, component], like=field, dtype=float, strided=True
        ) as column:
            scatter_linear_1d(
                positions[0],
                values[component],
                column,
                *args,
                shared_mem=8 * num_x,
                **launch,
            )


def _rows(field: Array) -> Array:
    """``field`` ``(nx, ny, nz, ncomp)`` as the rows ``(nx * ny * nz, ncomp)`` of the 3D kernels."""
    if not field.flags.c_contiguous:
        raise ValueError("the linear 3D kernels need a C-contiguous field")
    return field.reshape(-1, field.shape[-1])  # a view


def _scatter_linear_3d(
    positions: Array, values: Array, field: Array, *geometry: float | int
) -> None:
    """LINEAR's 3D deposit: the compiled kernel on the grid's rows."""
    scatter_linear_3d(positions, values, _rows(field), *geometry)


def _gather_linear_3d(
    positions: Array, field: Array, result: Array, *geometry: float | int
) -> None:
    """LINEAR's 3D gather: the compiled kernel on the grid's rows."""
    gather_linear_3d(positions, _rows(field), result, *geometry)


#: Linear (cloud-in-cell) weights: each marker on the two nearest nodes of each
#: axis. On a bounded axis a marker outside the grid uses the edge cell, its
#: weight clamped to [0, 1]; on a periodic one it wraps.
LINEAR = Weighting(
    "linear",
    scatter={1: _scatter_linear_1d, 2: scatter_linear_2d, 3: _scatter_linear_3d},
    gather={1: gather_linear_1d, 2: gather_linear_2d, 3: _gather_linear_3d},
    halo=1,
)

#: Quadratic B-spline weights ("triangular-shaped cloud"): each marker on its
#: nearest node and the two around it, with weights ``(1/2 - d)^2 / 2``,
#: ``3/4 - d^2`` and ``(1/2 + d)^2 / 2`` for the offset ``d`` (in cells) from the
#: nearest node. Smoother than :data:`LINEAR` (continuous first derivative, less
#: aliasing noise) for 3^ndim nodes per marker instead of 2^ndim. On a bounded
#: axis a position is clamped to the grid and a stencil node past the edge falls
#: on the edge node: the deposit keeps all the charge, the gather extrapolates flat.
QUADRATIC = Weighting(
    "quadratic",
    scatter={1: scatter_quadratic_1d, 2: scatter_quadratic_2d, 3: scatter_quadratic_3d},
    gather={1: gather_quadratic_1d, 2: gather_quadratic_2d, 3: gather_quadratic_3d},
    halo=2,
)
