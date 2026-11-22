"""Shape functions by B-spline degree: the node weighting and the edge weighting that go together.

The shapes of particle-in-cell are the uniform B-splines: degree 0 is the nearest grid
point, 1 cloud-in-cell (`LINEAR`), 2 the triangular-shaped cloud (`QUADRATIC`), 3 the
cubic spline (`CUBIC`). A `Splines` holds, for one degree p, the deposit and gather on
the nodes (the 0-forms, a `Weighting`) and on the edges (the 1-forms, an
`EdgeWeighting`) that match it. Matching means the derivative identity

    B_p'(x) = B_{p-1}(x + 1/2) - B_{p-1}(x - 1/2),

so that an edge function of direction a has degree p - 1 along a and degree p across,
and the divergence of the current deposited along a path is the change of the charge
deposited with the node weighting, exactly. `WHITNEY` is the edge weighting of degree 1,
`QUADRATIC_EDGES` and `CUBIC_EDGES` those of degrees 2 and 3 (a `Splines` of your own may
have none: ``edges`` is ``None``).

A code takes a `Splines`, or a degree resolved with `splines`, instead of importing the
weightings, and never pairs them itself::

    from mpifield import splines

    shape = splines(2)
    shape.nodes.scatter_block(positions, charges, rho, grid)
"""

from __future__ import annotations

from dataclasses import dataclass

from methodcard import method

from mpifield.edges import CUBIC_EDGES, QUADRATIC_EDGES, WHITNEY, EdgeWeighting
from mpifield.weighting import CUBIC, LINEAR, QUADRATIC, Weighting

__all__ = ["SPLINES", "Splines", "splines"]


@method(
    equations={
        r"derivative": r"B_p'(x) = B_{p-1}\!\left(x + \tfrac12\right) - B_{p-1}\!\left(x - \tfrac12\right)",
        r"edge_function": r"\mathbf w_{a,\mathbf k}(\mathbf x) = \mathbf e_a\, B_{p-1}\!\left(r_a - k_a - \tfrac12\right) \prod_{d \ne a} B_p(r_d - k_d)",
        r"continuity": r"\rho^{n+1}_{\mathbf k} - \rho^n_{\mathbf k} + (G^\top j)_{\mathbf k} = 0",
        r"halo": r"n_{\mathrm{halo}} = \max(n^{\mathrm{nodes}}_{\mathrm{halo}}, n^{\mathrm{edges}}_{\mathrm{halo}})",
    },
    assumptions=(
        "uniform B-splines of degree $p$ on a uniform grid, the same weights for a deposit onto node values (finite differences) and onto spline coefficients (finite elements)",
        "the edge weighting conserves the charge of the node weighting of the same degree; a code pairs them only through this object",
        "the built-in degrees 1 to 3 all have an edge weighting (`WHITNEY`, `QUADRATIC_EDGES`, `CUBIC_EDGES`); faces and cells (2- and 3-forms) are not there yet",
    ),
    references=(
        "[Kraus, Kormann, Morrison and Sonnendrücker, J. Plasma Phys. 83, 905830401 (2017)](https://doi.org/10.1017/S002237781700040X)",
        "[Esirkepov, Comput. Phys. Commun. 135, 144 (2001)](https://doi.org/10.1016/S0010-4655(00)00228-9)",
    ),
    components=(
        "mpifield.weighting.Weighting",
        "mpifield.edges.EdgeWeighting",
    ),
)
@dataclass(frozen=True, eq=False, repr=False)
class Splines:
    """The node weighting of B-spline degree ``degree`` and the edge weighting that matches it.

    ``nodes`` deposits and gathers on the nodes (0-forms), ``edges`` on the edges
    (1-forms), or is ``None`` while that degree has no edge weighting. Both must
    carry the degree.
    """

    degree: int
    nodes: Weighting
    edges: EdgeWeighting | None = None

    def __post_init__(self) -> None:
        """Check that the weightings have the degree."""
        if self.nodes.degree != self.degree:
            raise ValueError(
                f"the node weighting {self.nodes.name!r} has degree "
                f"{self.nodes.degree}, not {self.degree}"
            )
        if self.edges is not None and self.edges.degree != self.degree:
            raise ValueError(
                f"the edge weighting {self.edges.name!r} has degree "
                f"{self.edges.degree}, not {self.degree}"
            )

    @property
    def halo(self) -> int:
        """The halo nodes a split field needs for the node and the edge weighting."""
        edges = 0 if self.edges is None else self.edges.halo
        return max(self.nodes.halo, edges)

    def require_edges(self) -> EdgeWeighting:
        """Return the edge weighting, or raise if this degree has none yet."""
        if self.edges is None:
            have = ", ".join(str(s.degree) for s in SPLINES.values() if s.edges)
            raise ValueError(
                f"degree {self.degree} has no edge weighting yet (degrees with one: {have})"
            )
        return self.edges

    def __repr__(self) -> str:
        """Name the degree and the weightings."""
        edges = None if self.edges is None else self.edges.name
        return (
            f"Splines(degree={self.degree}, nodes={self.nodes.name!r}, edges={edges!r})"
        )


#: The built-in shapes, by degree.
SPLINES: dict[int, Splines] = {
    1: Splines(1, LINEAR, WHITNEY),
    2: Splines(2, QUADRATIC, QUADRATIC_EDGES),
    3: Splines(3, CUBIC, CUBIC_EDGES),
}


def splines(degree: int | Splines) -> Splines:
    """Return the built-in `Splines` of B-spline degree ``degree`` (a `Splines` is returned as is).

    Raises:
        ValueError: If there is no built-in shape of that degree.
    """
    if isinstance(degree, Splines):
        return degree
    if isinstance(degree, bool) or int(degree) != degree:
        raise ValueError(f"a spline degree is an integer, got {degree!r}")
    try:
        return SPLINES[int(degree)]
    except KeyError:
        raise ValueError(
            f"no built-in splines of degree {degree}; degrees {tuple(SPLINES)}"
        ) from None
