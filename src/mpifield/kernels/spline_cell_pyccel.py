"""The stencil of a position along one grid axis for B-spline weights of degree 0 to 3.

The Pyccel version of ``spline_cell.cuh`` (CUDA) and ``spline_cell.py`` (NumPy), shared
by the Pyccel spline kernels (``scatter_spline_*``, ``gather_spline_*``). A Pyccel kernel
imports it::

    from mpifield.kernels.spline_cell_pyccel import spline_cell

and cunumpy's kernel cache (``PyccelCache``) builds this module with the kernel.
``rel`` is the position in units of cells from the first node.

A plain function, not ``@inline``: it has local variables and a loop, and the
kernels call it from chunk functions whose calls are per thread, so its locals are
private to the thread. Comparisons instead of ``min``/``max``, which Pyccel turns
into calls. A ``periodic`` code of 2 marks a clamped axis (`clamped_cell`).
"""

import numpy as np
from pyccel.decorators import stack_array


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


def spline_cell(
    rel: float, num: int, periodic: int, degree: int, node: "int[:]", weight: "float[:]"
) -> None:
    """Fill the ``degree + 1`` stencil nodes and weights of ``rel`` along one axis.

    The stencil of an even degree is centred on the nearest node, the one of an odd
    degree on the cell holding the position; ``offset`` is the position from the
    middle node (``[-1/2, 1/2]`` for even, ``[0, 1)`` for odd degrees). On a periodic
    axis the ``num`` nodes cover ``num`` cells and the position wraps; on a bounded
    one it is clamped to ``[0, num - 1]`` and a stencil node past the edge is the
    edge node.
    """
    if periodic == 2:
        clamped_cell(rel, num, degree, 0, node, weight)
        return
    r = rel
    if periodic == 1:
        if not (r >= 0.0 and r < num):
            r = r - num * (r // num)
    else:
        if r < 0.0:
            r = 0.0
        if r > num - 1:
            r = num - 1.0
    middle = int(r + 0.5) if degree % 2 == 0 else int(r)
    offset = r - middle
    if degree == 0:
        weight[0] = 1.0
    elif degree == 1:
        weight[0] = 1.0 - offset
        weight[1] = offset
    elif degree == 2:
        weight[0] = 0.5 * (0.5 - offset) * (0.5 - offset)
        weight[1] = 0.75 - offset * offset
        weight[2] = 0.5 * (0.5 + offset) * (0.5 + offset)
    else:
        complement = 1.0 - offset
        square = offset * offset
        cube = square * offset
        weight[0] = complement * complement * complement / 6.0
        weight[1] = (3.0 * cube - 6.0 * square + 4.0) / 6.0
        weight[2] = (-3.0 * cube + 3.0 * square + 3.0 * offset + 1.0) / 6.0
        weight[3] = cube / 6.0
    first = middle - degree // 2
    for a in range(degree + 1):
        k = first + a
        if periodic == 1:
            k = (k + num) % num
        else:
            if k < 0:
                k = 0
            if k > num - 1:
                k = num - 1
        node[a] = k
