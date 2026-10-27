"""The stencil of positions along one grid axis for quadratic weights, vectorized.

The NumPy ``quadratic_cell.cuh``, shared by the NumPy versions of the quadratic
deposit and gather kernels, with the same arithmetic as their Pyccel and CUDA
versions.
"""

from __future__ import annotations

import numpy as np


def quadratic_cell(
    rel: np.ndarray, num_nodes: int, periodic: bool
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Return the three stencil nodes and their quadratic B-spline weights.

    ``rel`` is the position in units of the node spacing from the first node.
    The stencil is the nearest node ``c`` and its two neighbours; with the
    offset ``d = rel - c`` in [-1/2, 1/2] the weights are ``(1/2 - d)^2 / 2``,
    ``3/4 - d^2`` and ``(1/2 + d)^2 / 2``. On a periodic axis the ``num_nodes``
    nodes cover ``num_nodes`` cells and a position is wrapped into the period.
    On a bounded axis a position is clamped to [0, ``num_nodes`` - 1] and a
    stencil node past the edge is the edge node.
    """
    if periodic:
        rel = np.mod(rel, num_nodes)
        centre = np.floor(rel + 0.5).astype(np.int_)
        nodes = (
            (centre + num_nodes - 1) % num_nodes,
            centre % num_nodes,
            (centre + 1) % num_nodes,
        )
    else:
        rel = np.clip(rel, 0.0, num_nodes - 1)
        centre = np.floor(rel + 0.5).astype(np.int_)
        nodes = (
            np.maximum(centre - 1, 0),
            centre,
            np.minimum(centre + 1, num_nodes - 1),
        )
    offset = rel - centre
    weights = (
        0.5 * (0.5 - offset) ** 2,
        0.75 - offset**2,
        0.5 * (0.5 + offset) ** 2,
    )
    return nodes, weights
