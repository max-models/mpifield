"""Deposit and gather of k-forms of B-spline degree p in 1D to 3D, vectorized: the NumPy reference.

The spline de Rham sequence of degree p (`mpifield.splines`): a k-form component is the
tensor product of ``B_p(r_d - k_d)`` along the axes it is not staggered on and
``B_{p-1}(r_d - k_d - 1/2)`` along those it is: edges (k = 1) are staggered along their
direction, faces (k = 2 in 3D) across their normal, cells (k = ndim) along every axis.
Component ``c`` of a 1-form is the edge along axis ``c``, of a 2-form in 3D the face
normal to axis ``c``; the top form has one component. The values are integrals over the
k-cells (line, area, volume); these kernels neither divide by nor multiply with the
measures, so the scatter and the gather are each other's transpose.

Bounded axes as the node and edge weightings: positions clamped, the ``B_p`` factors fold
onto the edge node, the staggered ``B_{p-1}`` factors drop a k-cell past the wall. With
that rule the gathered fields commute with the discrete derivatives: the faces of
``C e`` gather the curl of the edge interpolant of ``e``, the cells of ``D f`` its
divergence. A deposit of the top form near a bounded wall therefore loses the share of
the dropped cells.
"""

from __future__ import annotations

import numpy as np

from mpifield.kernels.spline_cell import spline_cell
from mpifield.kernels.spline_edges import _check, _clamp, along


def staggered(form: int, ndim: int, component: int, axis: int) -> bool:
    """Whether component ``component`` of a ``form``-form is staggered along ``axis``."""
    if form == ndim:
        return True
    if form == 1:
        return axis == component
    if form == 2 and ndim == 3:
        return axis != component
    return False


def num_components(form: int, ndim: int) -> int:
    """The components of a ``form``-form in ``ndim`` dimensions (``ndim`` choose ``form``)."""
    return {(1, 1): 1, (1, 2): 2, (2, 2): 1, (1, 3): 3, (2, 3): 3, (3, 3): 1}[
        (form, ndim)
    ]


def _check_form(form: int, ndim: int, components: int) -> None:
    """Raise for a form the kernels do not cover or the wrong number of components."""
    if not 1 <= form <= ndim:
        raise ValueError(f"forms of {ndim} dimensions are 1 to {ndim}, got {form}")
    if components != num_components(form, ndim):
        raise ValueError(
            f"a {form}-form in {ndim}D has {num_components(form, ndim)} components, "
            f"got {components}"
        )


def _stencil(rel, num, periodic, degree, form, component):
    """``(index tuple, weight, exists)`` for every k-cell of a component's stencil."""
    ndim = rel.shape[0]
    parts = []
    for d in range(ndim):
        if staggered(form, ndim, component, d):
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


def _relative(positions, lower, cell_size, num, periodic):
    lower = np.asarray(lower, dtype=float)[:, None]
    cell_size = np.asarray(cell_size, dtype=float)[:, None]
    return _clamp((positions - lower) / cell_size, np.asarray(num), periodic)


def gather_forms_spline(
    positions: np.ndarray,
    field: np.ndarray,
    result: np.ndarray,
    lower: np.ndarray,
    cell_size: np.ndarray,
    num: np.ndarray,
    periodic: np.ndarray,
    degree: int,
    form: int,
) -> None:
    """Write ``sum W f`` of the ``form``-form ``field`` (*num, ncomp) at ``positions`` into ``result``."""
    _check(degree)
    ndim = positions.shape[0]
    _check_form(form, ndim, field.shape[-1])
    periodic = np.asarray(periodic, dtype=int)
    rel = _relative(positions, lower, cell_size, num, periodic)
    for component in range(field.shape[-1]):
        total = np.zeros(positions.shape[1])
        for index, weight, exists in _stencil(
            rel, num, periodic, degree, form, component
        ):
            total += np.where(exists, weight * field[(*index, component)], 0.0)
        result[component] = total


def scatter_forms_spline(
    positions: np.ndarray,
    values: np.ndarray,
    field: np.ndarray,
    lower: np.ndarray,
    cell_size: np.ndarray,
    num: np.ndarray,
    periodic: np.ndarray,
    degree: int,
    form: int,
) -> None:
    """Add ``W values`` (ncomp, N) at ``positions`` to the ``form``-form ``field`` (*num, ncomp).

    A marker whose values are all zero deposits nothing and its position is not read.
    """
    _check(degree)
    ndim = positions.shape[0]
    _check_form(form, ndim, field.shape[-1])
    keep = np.any(values != 0.0, axis=0)
    positions, values = positions[:, keep], values[:, keep]
    periodic = np.asarray(periodic, dtype=int)
    rel = _relative(positions, lower, cell_size, num, periodic)
    for component in range(field.shape[-1]):
        for index, weight, exists in _stencil(
            rel, num, periodic, degree, form, component
        ):
            amount = np.where(exists, weight * values[component], 0.0)
            np.add.at(field, (*index, np.full(positions.shape[1], component)), amount)
