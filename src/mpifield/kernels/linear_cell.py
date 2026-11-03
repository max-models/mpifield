"""The cell of positions along one grid axis, vectorized: the NumPy ``linear_cell.cuh``.

Shared by the NumPy versions of the deposit and gather kernels, with the same
arithmetic as their Pyccel and CUDA versions.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from methodcard import method

from mpifield import _method_checks


@method(
    equations={
        r"shape": r"w_1(\xi) = \begin{cases} 1 - |\xi| & |\xi| \le 1 \\ 0 & \text{otherwise} \end{cases}, \qquad W_{\mathbf k}(\mathbf x) = \prod_d w_1(r_d - k_d), \qquad r_d = \frac{x_d - x^{\mathrm{lo}}_d}{h_d}",
        r"weights": r"k = \lfloor r \rfloor, \quad f = r - k \in [0, 1], \qquad W_k = 1 - f, \quad W_{k+1} = f",
        r"partition": r"\sum_k W_k = 1",
        r"first_moment": r"\sum_k W_k\, k = r",
        r"bounded": r"k = \mathrm{clamp}(\lfloor r \rfloor, 0, N - 2), \qquad f = \mathrm{clamp}(r - k, 0, 1)",
        r"periodic": r"r \leftarrow r \bmod N, \qquad k = \lfloor r \rfloor, \qquad k + 1 \to (k + 1) \bmod N",
    },
    order=2,
    assumptions=(
        "a uniform grid; the weights of several axes are the products of the one-dimensional ones (two nodes per axis, $2^n$ in $n$ dimensions)",
        "the same arithmetic as the Pyccel kernels (inline) and the CUDA ones (`linear_cell.cuh`); the rules are checked with the compiled gathers",
        "beyond a bounded edge all the weight is on the edge node: a deposit loses no charge, a gather returns the edge value",
        "the order is that of the gather of a smooth field in the node spacing $h$",
    ),
    references=(
        "[Birdsall and Langdon, Plasma Physics via Computer Simulation (1991), ch. 8](https://doi.org/10.1201/9781315275048)",
        "[Hockney and Eastwood, Computer Simulation Using Particles (1988), ch. 5](https://www.routledge.com/products/9781138413375)",
    ),
    symbolic=partial(_method_checks.shape_rules, "linear"),
    problem=partial(_method_checks.gather_problem, "linear"),
)
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
