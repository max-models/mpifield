"""Clamped B-splines on a uniform 1D grid: evaluation, mass matrices and the incidence.

The 1D spaces of spline finite elements (FEEC, e.g. GEMPIC) with the same functions as
the clamped spline kernels (`mpifield.splines` with ``clamped=True``): the knots are the
``num_nodes`` grid nodes, the end ones repeated ``degree + 1`` times, so

- form 0 (potentials, charges) has the ``num_nodes + degree - 1`` B-splines ``B_i`` of
  degree p;
- form 1 (fields as line integrals) has the ``num_nodes + degree - 2`` functions
  ``D_i / h`` with ``D_i = p / (t_(i+p) - t_i) B_(i,p-1)``, for which
  ``B_i' = (D_(i-1) - D_i) / h`` in this indexing: the derivative of a form-0 field
  ``sum phi_i B_i`` is the form-1 field of ``G phi`` (`incidence`), exactly.

Everything is NumPy and SciPy sparse; for the particle side (deposit on form 0, gather
from form 1) use the clamped spline weightings, which give the same functions.
"""

from __future__ import annotations

import numpy as np
from methodcard import method
from scipy import sparse

from mpifield.kernels.spline_cell import clamped_cell

__all__ = [
    "clamped_knots",
    "evaluate",
    "incidence",
    "mass_matrix",
    "num_functions",
    "quadrature",
]


def _check(degree: int, form: int, num_nodes: int) -> None:
    """Raise for a degree, form or grid the module does not cover."""
    if degree not in (1, 2, 3):
        raise ValueError(f"clamped spline degrees are 1, 2 and 3, got {degree}")
    if form not in (0, 1):
        raise ValueError(f"1D forms are 0 and 1, got {form}")
    if num_nodes < 2:
        raise ValueError(f"a grid needs at least 2 nodes, got {num_nodes}")


def clamped_knots(
    lower: float, cell_size: float, num_nodes: int, degree: int
) -> np.ndarray:
    """The clamped knot vector: the nodes, the first and last repeated ``degree + 1`` times."""
    _check(degree, 0, num_nodes)
    nodes = lower + cell_size * np.arange(num_nodes)
    return np.concatenate(
        [np.full(degree, nodes[0]), nodes, np.full(degree, nodes[-1])]
    )


def num_functions(num_nodes: int, degree: int, form: int = 0) -> int:
    """The number of functions of form ``form``: ``num_nodes + degree - 1 - form``."""
    _check(degree, form, num_nodes)
    return num_nodes + degree - 1 - form


@method(
    equations={
        "knots": r"t_k = x_0 + h\, \mathrm{clip}(k - p,\ 0,\ N - 1), \qquad k = 0, \dots, N + 2p - 1",
        "form_0": r"\phi_h(x) = \sum_{i=0}^{N + p - 2} \phi_i\, B_{i,p}(x)",
        "form_1": r"E_h(x) = \sum_{i=0}^{N + p - 3} e_i\, \frac{D_i(x)}{h}, \qquad D_i = \frac{p}{(t_{i+1+p} - t_{i+1})/h}\, B_{i+1,p-1}",
        "derivative": r"\phi_h' = \sum_i (G\phi)_i\, \frac{D_i}{h}, \qquad (G\phi)_i = \phi_{i+1} - \phi_i",
        "mass": r"(M_k)_{ij} = \int W^k_i W^k_j\, dx \quad \text{(Gauss-Legendre, } p + 1 \text{ points per cell: exact)}",
    },
    assumptions=(
        "a uniform grid of $N$ nodes and spacing $h$; degrees 1 to 3",
        "the functions of the clamped spline kernels (`clamped_cell`), checked against `scipy.interpolate.BSpline`",
        "points outside the grid take the value at the nearest end",
    ),
    references=(
        "[de Boor, A Practical Guide to Splines (2001)](https://doi.org/10.1007/978-1-4612-6333-3)",
        "[Kraus, Kormann, Morrison and Sonnendrücker, J. Plasma Phys. 83, 905830401 (2017)](https://doi.org/10.1017/S002237781700040X)",
    ),
    components=("mpifield.kernels.spline_cell.clamped_cell",),
)
def evaluate(
    x: np.ndarray,
    lower: float,
    cell_size: float,
    num_nodes: int,
    degree: int,
    form: int = 0,
) -> sparse.csr_matrix:
    """The functions of form ``form`` at the points ``x``: a ``(len(x), n)`` sparse matrix.

    Form 0: ``B_i(x)``; form 1: ``D_i(x) / h``, so that the matrix times the edge
    values (line integrals) is the field.
    """
    n = num_functions(num_nodes, degree, form)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    rel = (x - lower) / cell_size
    columns, weights = clamped_cell(rel, num_nodes, degree, bool(form))
    rows = np.repeat(np.arange(x.size), len(columns))
    values = np.stack(weights, axis=1).ravel()
    if form == 1:
        values = values / cell_size
    return sparse.csr_matrix(
        (values, (rows, np.stack(columns, axis=1).ravel())), shape=(x.size, n)
    )


def quadrature(
    lower: float, cell_size: float, num_nodes: int, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """The Gauss-Legendre points and weights of ``count`` points in every cell, flattened."""
    if count < 1:
        raise ValueError(f"a quadrature needs at least one point, got {count}")
    reference, reference_weights = np.polynomial.legendre.leggauss(count)
    starts = lower + cell_size * np.arange(num_nodes - 1)
    points = starts[:, None] + 0.5 * cell_size * (reference + 1.0)
    weights = np.broadcast_to(0.5 * cell_size * reference_weights, points.shape)
    return points.ravel(), weights.ravel().copy()


def mass_matrix(
    lower: float, cell_size: float, num_nodes: int, degree: int, form: int = 0
) -> sparse.csr_matrix:
    """``M_ij = integral of W_i W_j dx`` for the functions of form ``form`` (exact quadrature)."""
    points, weights = quadrature(lower, cell_size, num_nodes, degree + 1)
    values = evaluate(points, lower, cell_size, num_nodes, degree, form)
    return (values.T @ sparse.diags(weights) @ values).tocsr()


def incidence(num_nodes: int, degree: int) -> sparse.csr_matrix:
    """``G``, form 0 to form 1: ``(G phi)_i = phi_(i+1) - phi_i``, ``(n - 1, n)``."""
    n = num_functions(num_nodes, degree, 0)
    return sparse.diags(
        [-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)
    ).tocsr()
