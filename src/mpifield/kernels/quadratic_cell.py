"""The stencil of positions along one grid axis for quadratic weights, vectorized.

The NumPy ``quadratic_cell.cuh``, shared by the NumPy versions of the quadratic
deposit and gather kernels, with the same arithmetic as their Pyccel and CUDA
versions.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from methodcard import method

from mpifield import _method_checks


@method(
    equations={
        r"shape": r"w_2(\xi) = \begin{cases} \tfrac34 - \xi^2 & |\xi| \le \tfrac12 \\ \tfrac12 \left(\tfrac32 - |\xi|\right)^2 & \tfrac12 < |\xi| \le \tfrac32 \\ 0 & \text{otherwise} \end{cases}, \qquad W_{\mathbf k}(\mathbf x) = \prod_d w_2(r_d - k_d)",
        r"weights": r"c = \lfloor r + \tfrac12 \rfloor, \quad \delta = r - c \in [-\tfrac12, \tfrac12], \qquad W_{c-1} = \tfrac12 \left(\tfrac12 - \delta\right)^2, \quad W_c = \tfrac34 - \delta^2, \quad W_{c+1} = \tfrac12 \left(\tfrac12 + \delta\right)^2",
        r"partition": r"\sum_k W_k = 1",
        r"first_moment": r"\sum_k W_k\, k = r",
        r"bounded": r"r \leftarrow \mathrm{clamp}(r, 0, N - 1), \qquad \text{nodes } \big(\max(c - 1, 0),\ c,\ \min(c + 1, N - 1)\big), \qquad W_0 = \tfrac12 \left(\tfrac12 - r\right)^2 + \tfrac34 - r^2 \ \ (0 \le r < \tfrac12)",
        r"periodic": r"r \leftarrow r \bmod N, \qquad \text{nodes } (c - 1, c, c + 1) \bmod N",
    },
    order=2,
    assumptions=(
        "a uniform grid; the weights of several axes are the products of the one-dimensional ones (three nodes per axis, $3^n$ in $n$ dimensions)",
        "the same arithmetic as the Pyccel kernels (`_quadratic_cell`) and the CUDA ones (`quadratic_cell.cuh`); the rules are checked with the compiled gathers",
        "on a node the weights are $(1/8, 3/4, 1/8)$; on and beyond a bounded edge $(7/8, 1/8)$ on the two edge nodes, so a deposit loses no charge and a gather is constant beyond the edge",
        "within half a cell of a bounded edge node the fold breaks the first moment: a linear field is not gathered exactly there",
        "the order is that of the gather of a smooth field in the node spacing $h$ (the second moment of $w_2$ is not zero)",
    ),
    references=(
        "[Hockney and Eastwood, Computer Simulation Using Particles (1988), ch. 5](https://www.routledge.com/products/9781138413375)",
        "[Birdsall and Langdon, Plasma Physics via Computer Simulation (1991), ch. 8](https://doi.org/10.1201/9781315275048)",
    ),
    symbolic=partial(_method_checks.shape_rules, "quadratic"),
    problem=partial(_method_checks.gather_problem, "quadratic"),
)
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
