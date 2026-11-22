"""The stencil of positions along one grid axis for B-spline weights of degree 0 to 3, vectorized.

The NumPy ``spline_cell.cuh``, shared by the NumPy versions of the spline deposit and
gather kernels (``scatter_spline_*``, ``gather_spline_*``), with the same arithmetic as
their Pyccel (``spline_cell_pyccel.py``) and CUDA versions.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from methodcard import method

from mpifield import _method_checks

#: The degrees the spline kernels cover.
DEGREES = (0, 1, 2, 3)


def _weights(degree: int, offset: np.ndarray) -> tuple[np.ndarray, ...]:
    """The uniform B-spline weights of the stencil, from the offset to the middle node."""
    if degree == 0:
        return (np.ones_like(offset),)
    if degree == 1:
        return (1.0 - offset, offset)
    if degree == 2:
        return (
            0.5 * (0.5 - offset) * (0.5 - offset),
            0.75 - offset * offset,
            0.5 * (0.5 + offset) * (0.5 + offset),
        )
    complement = 1.0 - offset
    square = offset * offset
    cube = square * offset
    return (
        complement * complement * complement / 6.0,
        (3.0 * cube - 6.0 * square + 4.0) / 6.0,
        (-3.0 * cube + 3.0 * square + 3.0 * offset + 1.0) / 6.0,
        cube / 6.0,
    )


@method(
    equations={
        r"shape": r"w_p = \underbrace{w_0 * \cdots * w_0}_{p + 1}, \qquad w_0(\xi) = \begin{cases} 1 & |\xi| < \tfrac12 \\ 0 & \text{otherwise} \end{cases}, \qquad W_{\mathbf k}(\mathbf x) = \prod_d w_p(r_d - k_d)",
        r"stencil": r"m = \begin{cases} \lfloor r + \tfrac12 \rfloor & p \text{ even} \\ \lfloor r \rfloor & p \text{ odd} \end{cases}, \qquad \text{nodes } m - \lfloor p/2 \rfloor, \dots, m - \lfloor p/2 \rfloor + p, \qquad \delta = r - m",
        r"cubic": r"W = \left(\tfrac{(1-\delta)^3}{6},\ \tfrac{3\delta^3 - 6\delta^2 + 4}{6},\ \tfrac{-3\delta^3 + 3\delta^2 + 3\delta + 1}{6},\ \tfrac{\delta^3}{6}\right), \quad \delta \in [0, 1)",
        r"partition": r"\sum_k W_k = 1",
        r"first_moment": r"\sum_k W_k\, k = r \quad (p \ge 1)",
        r"bounded": r"r \leftarrow \mathrm{clamp}(r, 0, N - 1), \qquad \text{a node past the edge is the edge node}",
        r"periodic": r"r \leftarrow r \bmod N, \qquad \text{nodes} \bmod N",
    },
    order=2,
    assumptions=(
        "a uniform grid; the weights of several axes are the products of the one-dimensional ones ($(p + 1)^n$ nodes in $n$ dimensions)",
        "degree 1 gives the weights of `LINEAR`, degree 2 those of `QUADRATIC`, with the same arithmetic",
        "the same arithmetic as the Pyccel block (`spline_cell_pyccel.spline_cell`) and the CUDA one (`spline_cell.cuh`); the rules are checked with the compiled cubic gathers",
        "on and beyond a bounded edge the stencil folds onto the edge node, so a deposit loses no charge and a gather is constant beyond the edge; within the stencil's reach of a bounded edge the fold breaks the first moment",
        "the order is that of the gather of a smooth field in the node spacing $h$, 2 for every degree from 1 on (the second moment of $w_p$ is $(p + 1)/12$, not zero)",
    ),
    references=(
        "[Hockney and Eastwood, Computer Simulation Using Particles (1988), ch. 5](https://www.routledge.com/products/9781138413375)",
        "[Birdsall and Langdon, Plasma Physics via Computer Simulation (1991), ch. 8](https://doi.org/10.1201/9781315275048)",
    ),
    symbolic=partial(_method_checks.shape_rules, "cubic"),
    problem=partial(_method_checks.gather_problem, "cubic"),
)
def spline_cell(
    rel: np.ndarray, num_nodes: int, periodic: bool, degree: int
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Return the ``degree + 1`` stencil nodes and their B-spline weights of degree ``degree``.

    ``rel`` is the position in units of the node spacing from the first node. The
    stencil of an even degree is centred on the nearest node, the one of an odd degree
    on the cell holding the position. On a periodic axis the ``num_nodes`` nodes cover
    ``num_nodes`` cells and a position is wrapped into the period. On a bounded axis a
    position is clamped to [0, ``num_nodes`` - 1] and a stencil node past the edge is
    the edge node.
    """
    if degree not in DEGREES:
        raise ValueError(f"spline degrees are {DEGREES}, got {degree}")
    rel = np.asarray(rel, dtype=float)
    if periodic:
        outside = ~((rel >= 0.0) & (rel < num_nodes))
        rel = np.where(outside, rel - num_nodes * np.floor(rel / num_nodes), rel)
    else:
        rel = np.clip(rel, 0.0, num_nodes - 1)
    shift = 0.5 if degree % 2 == 0 else 0.0
    middle = np.floor(rel + shift).astype(np.int_)
    first = middle - degree // 2
    nodes = []
    for a in range(degree + 1):
        k = first + a
        k = (k + num_nodes) % num_nodes if periodic else np.clip(k, 0, num_nodes - 1)
        nodes.append(k)
    return tuple(nodes), _weights(degree, rel - middle)
