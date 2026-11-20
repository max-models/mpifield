"""The weights of the lowest-order edge (Whitney) functions of a cell, for the Pyccel kernels.

The Pyccel version of ``whitney_cell.cuh`` (CUDA) and ``whitney_cell.py`` (NumPy). In a
cell the field of the line integrals ``e`` on its edges is, along each axis ``a``,

    E_a = sum over the cell's edges along a of  W(u, w) e / h_a,

with ``W`` the product of the linear weights across the cell (`edge_weight`) of the
fractions ``u``, ``w`` of the point along the other axes (one other axis in 2D, two in
3D; none in 1D, where ``W = 1``). A straight piece of path inside the cell adds
``q * (delta_a)`` times the average of ``W`` over the piece to each edge along ``a``
(``delta_a`` the piece's length along ``a`` in cells): in 2D the weight at the
piece's midpoint, in 3D `edge_path_weight`. A Pyccel kernel, also of another package,
imports the functions::

    from mpifield.kernels.whitney_cell_pyccel import edge_path_weight, edge_weight

and cunumpy's kernel cache (``PyccelCache``) builds this module with the kernel; Pyccel
inlines the calls. The functions have no local variables (see ``linear_cell_pyccel.py``).
"""

from pyccel.decorators import inline


@inline
def edge_weight(fraction: float, upper: int) -> float:
    """Return the linear weight of the lower (``upper`` 0) or upper (1) edge at ``fraction``."""
    return fraction if upper == 1 else 1.0 - fraction


@inline
def edge_path_weight(
    fraction_u: float,
    fraction_w: float,
    delta_u: float,
    delta_w: float,
    upper_u: int,
    upper_w: int,
) -> float:
    """Return the average of a 3D edge's bilinear weight over a straight piece.

    ``fraction_u`` and ``fraction_w`` are the piece's midpoint across the cell along the
    two other axes, ``delta_u`` and ``delta_w`` its lengths along them (in cells);
    ``upper_u`` and ``upper_w`` select the edge. The weight is quadratic along the piece:
    its average is the value at the midpoint plus ``+- delta_u delta_w / 12``.
    """
    return (fraction_u if upper_u == 1 else 1.0 - fraction_u) * (
        fraction_w if upper_w == 1 else 1.0 - fraction_w
    ) + (delta_u * delta_w / 12.0 if upper_u == upper_w else -delta_u * delta_w / 12.0)


@inline
def clamp_to_axis(rel: float, num: int, periodic: int) -> float:
    """Return ``rel`` on a bounded axis clamped to ``[0, num - 1]``; unchanged if periodic."""
    return (
        rel
        if periodic == 1
        else (0.0 if rel < 0.0 else (num - 1.0 if rel > num - 1.0 else rel))
    )


@inline
def plane_time(plane: float, rel: float, delta: float) -> float:
    """Return the time in ``[0, 1]`` units at which ``rel + t delta`` reaches ``plane``.

    ``2.0`` (never within the path) for ``delta`` 0.
    """
    return (plane - rel) / delta if delta != 0.0 else 2.0
