"""Edge gather and path deposit of B-spline degree p in 1D to 3D, vectorized: the NumPy reference.

The 1-forms that go with the node weighting of degree p (`mpifield.kernels.spline_cell`):
the function of the edge from node ``k`` along axis ``a`` is

    W(r) = B_{p-1}(r_a - k_a - 1/2) * prod_{b != a} B_p(r_b - k_b),

degree p - 1 along the edge, staggered by half a cell, and degree p across. Because
``B_p'(r) = B_{p-1}(r + 1/2) - B_{p-1}(r - 1/2)``, the divergence of the current deposited
along a path is the change of the node charge of degree p, exactly. Degree 1 gives the
Whitney functions of `mpifield.WHITNEY`.

Bounded axes, as the node weighting: positions and paths are clamped to the grid; across an
edge the stencil folds onto the edge node like the node stencil; along it an edge past the
wall is dropped, which is what the fold of the node charge leaves of it (the two nodes of
such an edge fold onto the same node). Periodic axes wrap. A path is cut where it crosses
a wall or a knot plane (integers for odd p, half-integers for even p), so every piece is
polynomial, and each piece is integrated exactly by Gauss-Legendre quadrature.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_cell import _weights, clamped_cell, spline_cell

#: The degrees the spline edge functions cover (the degree of the node weighting).
EDGE_DEGREES = (1, 2, 3)


def _check(degree: int) -> None:
    """Raise for a degree without spline edge functions."""
    if degree not in EDGE_DEGREES:
        raise ValueError(f"spline edge degrees are {EDGE_DEGREES}, got {degree}")


def along(
    rel: np.ndarray, num: int, periodic: bool, degree: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """The edges along an axis around ``rel``, their weights and whether each exists.

    ``rel`` is in cells from the first node and already clamped on a bounded axis.
    The weights are ``B_{p-1}(rel - k - 1/2)``; the ``degree`` edges are indexed by
    their first node. On a bounded axis the edges are ``0 .. num - 2``, on a periodic
    one ``0 .. num - 1`` (wrapped). On a clamped axis (``periodic`` 2) they are the
    rescaled clamped B-splines of `clamped_cell`, ``0 .. num + degree - 3``, all present.
    """
    if periodic == 2:
        edges, weights = clamped_cell(rel, num, degree, True)
        present = [np.ones(np.shape(rel), dtype=bool)] * degree
        return list(edges), list(weights), present
    shifted = rel - 0.5
    lower = degree - 1
    shift = 0.5 if lower % 2 == 0 else 0.0
    middle = np.floor(shifted + shift).astype(np.int_)
    if not periodic and lower % 2 == 0:
        # on the upper wall the stencil of the last cell (the degree-0 one jumps there)
        middle = np.minimum(middle, num - 2)
    first = middle - lower // 2
    weights = _weights(lower, shifted - middle)
    edges, exists = [], []
    for a in range(degree):
        k = first + a
        if periodic:
            edges.append(np.mod(k, num))
            exists.append(np.ones(k.shape, dtype=bool))
        else:
            edges.append(np.clip(k, 0, num - 2))
            exists.append((k >= 0) & (k <= num - 2))
    return edges, list(weights), exists


def _clamp(rel: np.ndarray, num: np.ndarray, periodic: np.ndarray) -> np.ndarray:
    """``rel`` (ndim, ...) clamped to ``[0, num - 1]`` on the bounded axes."""
    out = rel.copy()
    for d in range(rel.shape[0]):
        if periodic[d] != 1:
            out[d] = np.clip(out[d], 0.0, num[d] - 1)
    return out


def _edge_weights(
    rel: np.ndarray, num: np.ndarray, periodic: np.ndarray, degree: int, axis: int
):
    """The stencil of the edges along ``axis`` at the clamped positions ``rel`` (ndim, M).

    Yields ``(index tuple, weight, exists)`` for every edge of the stencil.
    """
    ndim = rel.shape[0]
    parts = []
    for d in range(ndim):
        if d == axis:
            parts.append(along(rel[d], int(num[d]), int(periodic[d]), degree))
        else:
            nodes, weights = spline_cell(rel[d], int(num[d]), int(periodic[d]), degree)
            parts.append(
                (list(nodes), list(weights), [np.ones(rel.shape[1], bool)] * len(nodes))
            )
    for combination in np.ndindex(*(len(p[0]) for p in parts)):
        index = tuple(parts[d][0][c] for d, c in enumerate(combination))
        weight = np.ones(rel.shape[1])
        exists = np.ones(rel.shape[1], dtype=bool)
        for d, c in enumerate(combination):
            weight = weight * parts[d][1][c]
            exists &= parts[d][2][c]
        yield index, weight, exists


def gather_edges_spline(
    positions: np.ndarray,
    edges: np.ndarray,
    result: np.ndarray,
    lower: np.ndarray,
    cell_size: np.ndarray,
    num: np.ndarray,
    periodic: np.ndarray,
    degree: int,
) -> None:
    """Write the field of ``edges`` (*num, ndim) at ``positions`` (ndim, N) into ``result``.

    ``E_a = sum_k W_{k,a} e_{k,a} / h_a``: the edge values are line integrals.
    """
    _check(degree)
    ndim = positions.shape[0]
    periodic = np.asarray(periodic, dtype=int)
    rel = _clamp(
        (positions - np.asarray(lower)[:, None]) / np.asarray(cell_size)[:, None],
        num,
        periodic,
    )
    for axis in range(ndim):
        total = np.zeros(positions.shape[1])
        for index, weight, exists in _edge_weights(rel, num, periodic, degree, axis):
            total += np.where(exists, weight * edges[(*index, axis)], 0.0)
        result[axis] = total / cell_size[axis]


#: Gauss-Legendre points on [0, 1] and their weights, by the number of points.
def _gauss(count: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(count)
    return 0.5 * (points + 1.0), 0.5 * weights


def _cuts(r0: np.ndarray, r1: np.ndarray, shift: float, num, periodic) -> np.ndarray:  # noqa: C901
    """The path parameters in (0, 1) where each path crosses a knot plane or a wall.

    ``r0`` and ``r1`` are ``(ndim, N)`` in cells; the result is ``(N, M)``, padded
    with 1.
    """
    ndim, count = r0.shape
    columns = []
    for d in range(ndim):
        offset = 0.0 if periodic[d] == 2 else shift  # clamped knots: the nodes
        low = np.minimum(r0[d], r1[d])
        high = np.maximum(r0[d], r1[d])
        first = np.floor(low - offset) + 1.0  # the first plane above low
        crossings = np.maximum(np.floor(high - offset) - first + 1.0, 0.0)
        most = int(crossings.max()) if count else 0
        delta = r1[d] - r0[d]
        safe = np.where(delta == 0.0, 1.0, delta)
        for j in range(most):
            plane = first + j + offset
            t = (plane - r0[d]) / safe
            columns.append(np.where(j < crossings, t, 1.0))
        if periodic[d] != 1:
            for wall in (0.0, num[d] - 1.0):
                t = (wall - r0[d]) / safe
                columns.append(np.where((delta != 0.0) & (t > 0.0) & (t < 1.0), t, 1.0))
    if not columns:
        return np.ones((count, 0))
    cuts = np.clip(np.stack(columns, axis=1), 0.0, 1.0)
    return np.sort(cuts, axis=1)


def deposit_paths_spline(
    start: np.ndarray,
    end: np.ndarray,
    charges: np.ndarray,
    edges: np.ndarray,
    lower: np.ndarray,
    cell_size: np.ndarray,
    num: np.ndarray,
    periodic: np.ndarray,
    degree: int,
) -> None:
    """Add the current of ``charges`` (N,) moving from ``start`` to ``end`` (ndim, N) to ``edges``.

    ``j_{k,a} += q * integral of W_{k,a} dr_a`` along the clamped path (``r`` in cells),
    piece by piece. A path with zero charge or a non-finite end deposits nothing.
    """
    _check(degree)
    ndim = start.shape[0]
    lower = np.asarray(lower, dtype=float)[:, None]
    cell_size = np.asarray(cell_size, dtype=float)[:, None]
    num = np.asarray(num)
    periodic = np.asarray(periodic, dtype=int)
    keep = (charges != 0.0) & np.all(np.isfinite(start), axis=0)
    keep &= np.all(np.isfinite(end), axis=0)
    r0 = (start[:, keep] - lower) / cell_size
    r1 = (end[:, keep] - lower) / cell_size
    q = charges[keep]
    if q.size == 0:
        return
    shift = 0.5 if degree % 2 == 0 else 0.0
    cuts = _cuts(r0, r1, shift, num, periodic)
    bounds = np.concatenate([np.zeros((q.size, 1)), cuts, np.ones((q.size, 1))], axis=1)
    # the integrand of a piece has degree (p - 1) + (ndim - 1) p
    points, gauss_weights = _gauss((degree - 1 + (ndim - 1) * degree) // 2 + 1)
    for piece in range(bounds.shape[1] - 1):
        t0, t1 = bounds[:, piece], bounds[:, piece + 1]
        length = t1 - t0
        if not np.any(length > 0.0):
            continue
        a0 = _clamp(r0 + t0 * (r1 - r0), num, periodic)
        a1 = _clamp(r0 + t1 * (r1 - r0), num, periodic)
        step = a1 - a0  # zero across a wall, where the clamped path stands still
        for point, gauss_weight in zip(points, gauss_weights, strict=True):
            rel = a0 + point * step
            for axis in range(ndim):
                amount = q * gauss_weight * step[axis]
                if not np.any(amount != 0.0):
                    continue
                for index, weight, exists in _edge_weights(
                    rel, num, periodic, degree, axis
                ):
                    contribution = np.where(exists, weight * amount, 0.0)
                    np.add.at(edges, (*index, np.full(q.size, axis)), contribution)
