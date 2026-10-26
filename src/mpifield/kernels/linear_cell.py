"""The cell of positions along one grid axis, vectorized: the NumPy ``linear_cell.cuh``.

Shared by the NumPy versions of the deposit and gather kernels, with the same
arithmetic as their Pyccel and CUDA versions.
"""

from __future__ import annotations

import numpy as np


def linear_cell(
    rel: np.ndarray, num_nodes: int, periodic: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(left, right, fraction)`` for linear weights along one axis.

    ``rel`` is the position in units of the node spacing from the first node.
    On a periodic axis the ``num_nodes`` nodes cover ``num_nodes`` cells and
    the last node's right neighbour is the first node; a position is wrapped
    into the period.  On a bounded axis the indices are clamped to the last
    cell, so a point outside the grid uses the edge node.  The fraction (the
    weight of the right node) is clamped to [0, 1].
    """
    if periodic:
        rel = np.mod(rel, num_nodes)
        left = np.mod(np.floor(rel).astype(np.int_), num_nodes)
        right = (left + 1) % num_nodes
        fraction = np.clip(rel - left, 0.0, 1.0)
        return left, right, fraction
    left = np.clip(np.floor(rel).astype(np.int_), 0, num_nodes - 2)
    fraction = np.clip(rel - left, 0.0, 1.0)
    return left, left + 1, fraction
