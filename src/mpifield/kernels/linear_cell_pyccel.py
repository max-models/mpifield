"""The linear cell of a position along one grid axis, for the Pyccel kernels.

The Pyccel version of ``linear_cell.cuh`` (CUDA) and ``linear_cell.py`` (NumPy): the left
and right node of the cell holding the position and the weight of the right node, on a
bounded axis (positions clamped to the first and last cell) or a periodic one (the
``num`` nodes cover ``num`` cells, the last node neighbours the first). A Pyccel kernel,
also of another package, imports the functions::

    from mpifield.kernels.linear_cell_pyccel import cell_fraction, left_node

and cunumpy's kernel cache (``PyccelCache``) builds this module with the kernel; Pyccel
inlines the calls. ``rel`` is the position in units of cells from the first node.

The functions have no local variables: inlined into a loop under
``#$ omp parallel for``, Pyccel would declare them outside the loop, shared by the
threads. Comparisons instead of ``min``/``max``, which Pyccel turns into calls.
"""

from pyccel.decorators import inline


@inline
def periodic_position(rel: float, num: int) -> float:
    """Return ``rel`` on a periodic axis of ``num`` cells, in ``[0, num)``.

    The floor modulo is taken only for a position outside ``[0, num)``.
    """
    return rel if rel >= 0.0 and rel < num else rel - num * (rel // num)


@inline
def left_node(rel: float, num: int, periodic: int) -> int:
    """Return the left node of the cell holding ``rel`` (after `periodic_position`).

    Bounded (``periodic`` 0): clamped to ``0 .. num - 2``; periodic: ``0 .. num - 1``.
    """
    return (
        (num - 1 if int(rel) >= num else int(rel))
        if periodic == 1
        else (
            num - 2
            if (int(rel) if rel >= 0.0 else 0) > num - 2
            else (int(rel) if rel >= 0.0 else 0)
        )
    )


@inline
def right_node(left: int, num: int, periodic: int) -> int:
    """Return the right node of the cell: ``left + 1``, wrapped to 0 on a periodic axis."""
    return 0 if periodic == 1 and left + 1 == num else left + 1


@inline
def cell_fraction(rel: float, left: int) -> float:
    """Return the weight of the right node, ``rel - left`` clamped to ``[0, 1]``."""
    return 0.0 if rel - left < 0.0 else (1.0 if rel - left > 1.0 else rel - left)
