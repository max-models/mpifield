"""Edge fields on a block: the lowest-order (Whitney) edge gather and the path deposit.

An edge field holds the line integral of a vector field along every edge of a uniform
grid, as an array ``num_nodes + (ndim,)``: ``edges[..., a]`` is the integral along the
edge from a node to its neighbour along axis ``a`` (on a periodic axis the last node's
edge goes to the first node; on a bounded axis the last node has no edge along it and
its entry stays zero). Inside a cell the field is the lowest-order Nédélec (Whitney)
interpolant: the component along an axis is multilinear across the cell between the
cell's edges along that axis, and constant along them. `WHITNEY` pairs the two
operations on it, compiled like `LINEAR`'s:

- `EdgeWeighting.gather_block`: the field of an edge field at the markers;
- `EdgeWeighting.deposit_paths_block`: the current of charges moving along straight
  paths, the line integral of every edge function along them; its transpose is the
  work of the field on the charges.

With the discrete gradient ``G`` (`EdgeWeighting.gradient_block`: the potential's
difference along every edge) the deposit conserves charge exactly: ``G^T j``
(`EdgeWeighting.divergence_block`) is `LINEAR`'s deposit of the charges at the ends of
their paths minus at the starts. This is the coupling of an energy- and
charge-conserving implicit PIC; patchwork composes the block kernels across patches.

The kernel convention, as `mpifield.weighting`'s::

    gather(positions, edges, result, *geometry)              # overwrites result
    deposit_paths(start, end, charges, edges, *geometry)      # adds into edges

with ``geometry = (*lower, *cell_size, *num_nodes, *periodic)`` (`BlockGrid.geometry`),
``positions``, ``start``, ``end`` and ``result`` ``(ndim, N)`` and ``charges``
``(N,)``. Paths are not wrapped on a periodic axis; on a bounded axis they are clamped
to the grid, as `LINEAR` clamps a position.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any

import cunumpy as xp
from methodcard import method

from mpifield import _method_checks
from mpifield._types import Array, BlockGrid
from mpifield.kernels.deposit_paths_1d import kernel as deposit_paths_1d
from mpifield.kernels.deposit_paths_2d import kernel as deposit_paths_2d
from mpifield.kernels.deposit_paths_3d import kernel as deposit_paths_3d
from mpifield.kernels.gather_edges_1d import kernel as gather_edges_1d
from mpifield.kernels.gather_edges_2d import kernel as gather_edges_2d
from mpifield.kernels.gather_edges_3d import kernel as gather_edges_3d

__all__ = ["WHITNEY", "EdgeWeighting"]

KernelFunction = Callable[..., Any]


def _checked(kernels: Mapping[int, KernelFunction], role: str) -> Mapping[int, Any]:
    """Return ``kernels`` as a read-only mapping, checking the dimensions."""
    for ndim in kernels:
        if ndim not in (1, 2, 3):
            raise ValueError(f"{role} kernels are for 1, 2 or 3 dimensions, got {ndim}")
    return MappingProxyType(dict(kernels))


@method(
    equations={
        "whitney": r"E_a(\mathbf x) = \frac{1}{h_a} \sum_{\text{edges } \epsilon \parallel a} W_\epsilon(\mathbf x)\, e_\epsilon, \qquad W_\epsilon = \prod_{b \ne a} w_1(r_b - k_b)",
        "path": r"j_\epsilon \mathrel{+}= q \int_{\mathbf x_0}^{\mathbf x_1} \frac{W_\epsilon}{h_a}\, dx_a",
        "continuity": r"G^\top j = \rho(\mathbf x_1) - \rho(\mathbf x_0) \quad \text{(LINEAR's deposit)}",
        "work": r"j^\top e = \sum_k q_k \int_{\mathbf x_0}^{\mathbf x_1} \mathbf E \cdot d\mathbf l",
        "gradient": r"(G\phi)_\epsilon = \phi_{\mathrm{end}(\epsilon)} - \phi_{\mathrm{start}(\epsilon)}, \qquad \mathbf E(G\phi) = \nabla (\text{multilinear } \phi)",
    },
    assumptions=(
        "a uniform grid; an edge field holds line integrals, `num_nodes + (ndim,)`",
        "straight paths, cut at every grid line or plane they cross; each piece lies in one cell and is integrated exactly (the midpoint in 2D, the midpoint plus the product of the two other lengths over 12 in 3D)",
        "on a bounded axis paths and positions are clamped to the grid, as LINEAR clamps a position: the current is that of LINEAR's charge, and the clamped ends move along the wall",
        "paths are not wrapped on a periodic axis; the seam's edges are those of the last node",
        "a path with zero charge or a non-finite end deposits nothing",
    ),
    references=(
        "[Bossavit, IEE Proc. A 135, 493 (1988)](https://doi.org/10.1049/ip-a-1.1988.0077)",
        "[Villasenor and Buneman, Comput. Phys. Commun. 69, 306 (1992)](https://doi.org/10.1016/0010-4655(92)90169-Y)",
        "[Chen, Chacón and Barnes, J. Comput. Phys. 230, 7018 (2011)](https://doi.org/10.1016/j.jcp.2011.05.031)",
    ),
    symbolic=_method_checks.edge_rules,
)
@dataclass(frozen=True, eq=False, repr=False)
class EdgeWeighting:
    """The edge gather and its path deposit, for each number of dimensions they cover.

    ``gather`` and ``deposit_paths`` map a number of dimensions (1, 2, 3) to a kernel
    with the arguments of the module's convention. ``halo`` is the number of halo nodes
    a split edge field needs so that every path piece in a block's own cells deposits
    into the stored block: 1, the cell's upper nodes. ``degree`` is the B-spline degree
    of the node weighting whose charge the path deposit conserves (``None`` when it is
    not one).
    """

    name: str
    gather: Mapping[int, KernelFunction]
    deposit_paths: Mapping[int, KernelFunction]
    halo: int = 1
    degree: int | None = None

    def __post_init__(self) -> None:
        """Check the dimensions, the halo and the degree; freeze the kernel mappings."""
        if self.halo < 0:
            raise ValueError(f"halo must be >= 0, got {self.halo}")
        if self.degree is not None and self.degree < 1:
            raise ValueError(f"degree must be >= 1, got {self.degree}")
        object.__setattr__(self, "gather", _checked(self.gather, "gather"))
        object.__setattr__(
            self, "deposit_paths", _checked(self.deposit_paths, "deposit_paths")
        )

    def _kernel(self, kernels: Mapping[int, KernelFunction], ndim: int, role: str):
        """Return the kernel for ``ndim`` dimensions, or raise naming the weighting."""
        try:
            return kernels[ndim]
        except KeyError:
            have = ", ".join(f"{n}D" for n in sorted(kernels)) or "none"
            raise ValueError(
                f"the {self.name} edge weighting has no {ndim}D {role} kernel (has {have})"
            ) from None

    def gather_kernel(self, ndim: int) -> KernelFunction:
        """Return the gather kernel for ``ndim`` dimensions."""
        return self._kernel(self.gather, ndim, "gather")

    def deposit_paths_kernel(self, ndim: int) -> KernelFunction:
        """Return the path deposit kernel for ``ndim`` dimensions."""
        return self._kernel(self.deposit_paths, ndim, "deposit_paths")

    @staticmethod
    def _check(edges: Array, grid: BlockGrid) -> int:
        """The number of dimensions of ``grid``, checking the shape of ``edges``."""
        ndim = len(grid.periodic)
        expected = (*(int(n) for n in grid.num_gridpoints), ndim)
        if tuple(edges.shape) != expected:
            raise ValueError(
                f"an edge field on this grid is {expected}, got {edges.shape}"
            )
        return ndim

    def gather_block(
        self, positions: Array, edges: Array, result: Array, grid: BlockGrid
    ) -> None:
        """Write the field of ``edges``, a block on ``grid``, at ``positions`` into ``result``.

        ``positions`` and ``result`` are ``(ndim, N)``; ``result`` may be on either side
        of ``edges``, and the kernel runs on the side of ``edges``.
        """
        kernel = self.gather_kernel(self._check(edges, grid))
        positions = xp.kernels.as_kernel_array(
            positions, like=edges, dtype=float, strided=True
        )
        values = xp.kernels.as_kernel_array(edges, like=edges, dtype=float)
        with xp.kernels.kernel_output(
            result, like=edges, dtype=float, strided=True
        ) as out:
            kernel(positions, values, out, *grid.geometry)

    def deposit_paths_block(
        self,
        start: Array,
        end: Array,
        charges: Array,
        edges: Array,
        grid: BlockGrid,
    ) -> None:
        """Add the current of ``charges`` moving from ``start`` to ``end`` to ``edges``.

        ``start`` and ``end`` are ``(ndim, N)`` (not wrapped on a periodic axis),
        ``charges`` ``(N,)``; ``edges`` is a block on ``grid``, on the CPU or the GPU,
        and the paths are moved to its side. Each edge gets ``q`` times the line
        integral of its edge function along the paths (in units of the cell size along
        it), so ``j . e`` is the work of the field of ``e`` and
        `divergence_block` of ``j`` the change of `LINEAR`'s charge.
        """
        kernel = self.deposit_paths_kernel(self._check(edges, grid))
        convert = partial(
            xp.kernels.as_kernel_array, like=edges, dtype=float, strided=True
        )
        start, end = convert(start), convert(end)
        charges = convert(charges).reshape(-1)
        if start.shape != end.shape or start.shape[-1] != charges.shape[0]:
            raise ValueError(
                f"paths from {start.shape} to {end.shape} for {charges.shape[0]} charges"
            )
        with xp.kernels.kernel_output(edges, like=edges, dtype=float) as buffer:
            kernel(start, end, charges, buffer, *grid.geometry)

    @staticmethod
    def gradient_block(nodes: Array, grid: BlockGrid) -> Array:
        """The discrete gradient ``G phi``: a node field's difference along every edge.

        ``nodes`` is ``num_nodes`` (one component); returns the edge field
        ``num_nodes + (ndim,)``, zero on the missing last edges of a bounded axis. The
        field of ``G phi`` is the gradient of `LINEAR`'s interpolant of ``phi``.
        """
        ndim = len(grid.periodic)
        lib = xp.get_array_module(nodes)
        result = lib.zeros((*nodes.shape, ndim))
        for axis, periodic in enumerate(grid.periodic):
            if periodic:
                result[..., axis] = lib.roll(nodes, -1, axis=axis) - nodes
            else:
                lower = [slice(None)] * ndim
                upper = [slice(None)] * ndim
                lower[axis] = slice(0, -1)
                upper[axis] = slice(1, None)
                result[(*lower, axis)] = nodes[tuple(upper)] - nodes[tuple(lower)]
        return result

    @staticmethod
    def divergence_block(edges: Array, grid: BlockGrid) -> Array:
        """The transpose of `gradient_block`, ``G^T j``: ``num_nodes`` from an edge field.

        Of a deposited current, the change of `LINEAR`'s charge on every node.
        """
        ndim = len(grid.periodic)
        lib = xp.get_array_module(edges)
        result = lib.zeros(edges.shape[:-1])
        for axis, periodic in enumerate(grid.periodic):
            flow = edges[..., axis]
            if periodic:
                result += lib.roll(flow, 1, axis=axis) - flow
            else:
                lower = [slice(None)] * ndim
                upper = [slice(None)] * ndim
                lower[axis] = slice(0, -1)
                upper[axis] = slice(1, None)
                result[tuple(lower)] -= flow[tuple(lower)]
                result[tuple(upper)] += flow[tuple(lower)]
        return result

    def __repr__(self) -> str:
        """Name the weighting, its dimensions and its halo."""
        dims = sorted(set(self.gather) | set(self.deposit_paths))
        return f"EdgeWeighting({self.name!r}, ndim={tuple(dims)}, halo={self.halo})"


#: The lowest-order edge (Whitney, Nédélec first kind) gather and its path deposit,
#: in 1D, 2D and 3D: the edge space whose node space is `LINEAR`'s.
WHITNEY = EdgeWeighting(
    "whitney",
    gather={1: gather_edges_1d, 2: gather_edges_2d, 3: gather_edges_3d},
    deposit_paths={1: deposit_paths_1d, 2: deposit_paths_2d, 3: deposit_paths_3d},
    halo=1,
    degree=1,
)
