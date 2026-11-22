"""The edge stencil, the path cuts and the Gauss rules of the spline edge kernels, for Pyccel.

The Pyccel version of ``spline_edges.cuh`` (CUDA) and of the helpers of
``spline_edges.py`` (NumPy), shared by ``gather_edges_spline_*`` and
``deposit_paths_spline_*``::

    from mpifield.kernels.spline_edges_pyccel import along_stencil, gauss_rule, path_cuts

Plain functions (locals, loops): the kernels call them from chunk functions whose calls
are per thread. Comparisons instead of ``min``/``max``, which Pyccel turns into calls.
A ``periodic`` code of 2 marks a clamped axis (`clamped_cell`, a copy of the one in
``spline_cell_pyccel``: the kernels import from one block module each).
"""

import numpy as np
from pyccel.decorators import stack_array


def floor_int(x: float) -> int:
    """The largest integer not above ``x``."""
    whole = int(x)
    if x < whole:
        whole -= 1
    return whole


@stack_array("left", "right")
def clamped_cell(
    rel: float,
    num: int,
    degree: int,
    staggered: int,
    node: "int[:]",
    weight: "float[:]",
) -> None:
    """The clamped B-spline stencil of ``rel`` (in cells) on an axis of ``num`` nodes.

    The knots are the nodes, the end ones repeated ``degree + 1`` times
    (``t_k = clip(k - degree, 0, num - 1)``); there are ``num + degree - 1`` functions.
    With ``staggered`` 0 these are the ``degree + 1`` B-splines of degree ``degree``
    that are non-zero in the cell of ``rel``, indexed from the first one; with 1 the
    ``degree`` rescaled B-splines ``D_i = p / (t_(i+p) - t_i) B_(i,p-1)`` (``i >= 1``),
    indexed ``i - 1``, for which ``B_(i,p)' = D_i - D_(i+1)``. Cox-de Boor recursion.
    """
    left = np.zeros(4)
    right = np.zeros(4)
    r = rel
    if r < 0.0:
        r = 0.0
    if r > num - 1:
        r = num - 1.0
    cell = int(r)
    if cell > num - 2:
        cell = num - 2
    span = cell + degree
    order = degree - staggered
    weight[0] = 1.0
    for k in range(1, order + 1):
        low = span + 1 - k - degree
        high = span + k - degree
        if low < 0:
            low = 0
        if high > num - 1:
            high = num - 1
        left[k] = r - low
        right[k] = high - r
        saved = 0.0
        for m in range(k):
            temp = weight[m] / (right[m + 1] + left[k - m])
            weight[m] = saved + right[m + 1] * temp
            saved = left[k - m] * temp
        weight[k] = saved
    for m in range(order + 1):
        node[m] = cell + m
        if staggered == 1:
            i = cell + 1 + m
            low = i - degree
            high = i
            if low < 0:
                low = 0
            if high > num - 1:
                high = num - 1
            weight[m] = weight[m] * degree / (high - low)


def along_stencil(
    rel: float,
    num: int,
    periodic: int,
    degree: int,
    edge: "int[:]",
    weight: "float[:]",
    exists: "int[:]",
) -> None:
    """The ``degree`` edges along an axis around ``rel`` (clamped), weights ``B_{p-1}(rel - k - 1/2)``.

    Edges are indexed by their first node: ``0 .. num - 2`` on a bounded axis (one past
    it does not exist), ``0 .. num - 1`` wrapped on a periodic one. On the upper wall
    the stencil is the last cell's.
    """
    if periodic == 2:
        clamped_cell(rel, num, degree, 1, edge, weight)
        for a in range(degree):
            exists[a] = 1
        return
    lower = degree - 1
    r = rel
    if periodic == 1 and not (r >= 0.0 and r < num):
        r = r - num * (r // num)
    shifted = r - 0.5
    if lower % 2 == 0:
        middle = floor_int(shifted + 0.5)
        if periodic == 0 and middle > num - 2:
            middle = num - 2
    else:
        middle = floor_int(shifted)
    offset = shifted - middle
    if lower == 0:
        weight[0] = 1.0
    elif lower == 1:
        weight[0] = 1.0 - offset
        weight[1] = offset
    else:
        weight[0] = 0.5 * (0.5 - offset) * (0.5 - offset)
        weight[1] = 0.75 - offset * offset
        weight[2] = 0.5 * (0.5 + offset) * (0.5 + offset)
    first = middle - lower // 2
    for a in range(degree):
        k = first + a
        if periodic == 1:
            edge[a] = (k + 2 * num) % num
            exists[a] = 1
        elif k < 0 or k > num - 2:
            edge[a] = 0
            exists[a] = 0
        else:
            edge[a] = k
            exists[a] = 1


def gauss_rule(count: int, point: "float[:]", weight: "float[:]") -> None:
    """The Gauss-Legendre rule of ``count`` (1 to 5) points on [0, 1]."""
    if count == 1:
        point[0] = 0.5
        weight[0] = 1.0
    elif count == 2:
        point[0] = 0.21132486540518713
        point[1] = 0.7886751345948129
        weight[0] = 0.5
        weight[1] = 0.5
    elif count == 3:
        point[0] = 0.1127016653792583
        point[1] = 0.5
        point[2] = 0.8872983346207417
        weight[0] = 0.2777777777777778
        weight[1] = 0.4444444444444444
        weight[2] = 0.2777777777777778
    elif count == 4:
        point[0] = 0.06943184420297371
        point[1] = 0.33000947820757187
        point[2] = 0.6699905217924281
        point[3] = 0.9305681557970262
        weight[0] = 0.17392742256872692
        weight[1] = 0.3260725774312731
        weight[2] = 0.3260725774312731
        weight[3] = 0.17392742256872692
    else:
        point[0] = 0.04691007703066802
        point[1] = 0.23076534494715845
        point[2] = 0.5
        point[3] = 0.7692346550528415
        point[4] = 0.9530899229693319
        weight[0] = 0.11846344252809454
        weight[1] = 0.2393143352496832
        weight[2] = 0.28444444444444444
        weight[3] = 0.2393143352496832
        weight[4] = 0.11846344252809454


def path_cuts(
    r0: "float[:]",
    r1: "float[:]",
    ndim: int,
    shift: float,
    num: "int[:]",
    periodic: "int[:]",
    cuts: "float[:]",
) -> int:
    """Fill ``cuts`` with 0, the sorted parameters in (0, 1) of knot planes and walls, and 1.

    A path from ``r0`` to ``r1`` (in cells) crosses the planes ``k + shift`` and, on a
    bounded axis, the walls 0 and ``num - 1``; returns the number of entries. The
    caller keeps every axis's extent below 4 cells, so ``cuts`` of 32 entries suffice.
    """
    count = 1
    cuts[0] = 0.0
    for d in range(ndim):
        delta = r1[d] - r0[d]
        if delta == 0.0:
            continue
        low = r0[d] if delta > 0.0 else r1[d]
        high = r1[d] if delta > 0.0 else r0[d]
        offset = 0.0 if periodic[d] == 2 else shift  # clamped knots: the nodes
        plane = floor_int(low - offset) + 1.0 + offset  # the first plane above low
        while plane < high:
            t = (plane - r0[d]) / delta
            if t > 0.0 and t < 1.0:
                cuts[count] = t
                count += 1
            plane += 1.0
        if periodic[d] != 1:
            t = (0.0 - r0[d]) / delta
            if t > 0.0 and t < 1.0:
                cuts[count] = t
                count += 1
            t = (num[d] - 1.0 - r0[d]) / delta
            if t > 0.0 and t < 1.0:
                cuts[count] = t
                count += 1
    cuts[count] = 1.0
    count += 1
    # insertion sort of the few cuts
    for i in range(2, count - 1):
        value = cuts[i]
        j = i - 1
        while j > 0 and cuts[j] > value:
            cuts[j + 1] = cuts[j]
            j -= 1
        cuts[j + 1] = value
    return count


def staggered(form: int, ndim: int, component: int, axis: int) -> int:
    """1 where component ``component`` of a ``form``-form is staggered along ``axis``.

    Edges along their direction, faces (2-forms in 3D) across their normal, the top
    form along every axis.
    """
    if form == ndim:
        return 1
    if form == 1:
        return 1 if axis == component else 0
    if form == 2 and ndim == 3:
        return 0 if axis == component else 1
    return 0
