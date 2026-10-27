"""Typing aliases and the small value types shared by mpifield's modules."""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias

import numpy as np
from numpy import ndarray

Array: TypeAlias = ndarray
Number: TypeAlias = int | float
PathLikeStr: TypeAlias = str | Path | PathLike[str]
IndexLike: TypeAlias = int | slice | tuple[int | slice, ...]
Kernel: TypeAlias = Callable[..., Any]


class Bounds(NamedTuple):
    """The lower and upper corner of a grid block, one entry per axis."""

    lower_bound: Array
    upper_bound: Array


class BlockGrid(NamedTuple):
    """The grid of one rank's stored block, as the kernels see it (:meth:`Field.block_grid`).

    ``geometry`` is the same grid as the kernels' trailing arguments, Python
    scalars: ``(*lower, *cell_size, *num_nodes, *periodic)``, one entry per axis
    each (``periodic`` as 0 or 1). ``index`` selects the nodes of the grid in
    :attr:`Field.local_with_halos`.
    """

    bounds: Bounds
    grid_spacing: Array
    num_gridpoints: Array
    periodic: tuple[bool, ...]
    geometry: tuple[float | int, ...]
    index: tuple[slice, ...]

    @classmethod
    def from_geometry(
        cls,
        lower: tuple[float, ...],
        cell_size: tuple[float, ...],
        num_nodes: tuple[int, ...],
        periodic: tuple[bool, ...] | None = None,
    ) -> BlockGrid:
        """Return the grid of a whole array of ``num_nodes`` nodes from ``lower`` on.

        For running a weighting on an array that is not a field's stored block
        (:meth:`Weighting.scatter_block`). The arguments may be sequences or
        arrays on either side (a :meth:`Field.block_grid` on the GPU holds CuPy
        arrays). ``periodic`` defaults to bounded axes.
        """
        lower = tuple(float(v) for v in lower)
        cell_size = tuple(float(v) for v in cell_size)
        num_nodes = tuple(int(n) for n in num_nodes)
        ndim = len(num_nodes)
        flags = (False,) * ndim if periodic is None else tuple(map(bool, periodic))
        if not len(lower) == len(cell_size) == len(flags) == ndim:
            raise ValueError(
                "lower, cell_size, num_nodes and periodic differ in length"
            )
        lower_bound = np.asarray(lower, dtype=float)
        spacing = np.asarray(cell_size, dtype=float)
        nodes = np.asarray(num_nodes, dtype=int)
        return cls(
            bounds=Bounds(
                lower_bound=lower_bound, upper_bound=lower_bound + (nodes - 1) * spacing
            ),
            grid_spacing=spacing,
            num_gridpoints=nodes,
            periodic=flags,
            geometry=(*lower, *cell_size, *num_nodes, *(int(flag) for flag in flags)),
            index=(slice(None),) * ndim,
        )
