"""The current of charges moving along straight paths, on the spline edge functions of degree 1 to 3, in 2D.

The Pyccel version (CPU) of the ``deposit_paths_spline_2d`` kernel; ``deposit_paths_spline_2d_cuda.cu``
next to this file is the CUDA version (GPU), with the same arguments. Each path, clamped
to the grid on bounded axes, is cut at the knot planes and walls (``spline_edges_pyccel``)
and every piece is integrated exactly by Gauss-Legendre quadrature:
``j_(k,a) += q * integral of W_(k,a) dr_a``. Serial (no OpenMP yet). A path with zero
charge or a non-finite end deposits nothing.

``start`` and ``end`` are ``(2, N)``, ``charges`` ``(N,)``, ``edges`` ``(*num, 2)``.
"""

import numpy as np

from mpifield.kernels.spline_cell_pyccel import spline_cell
from mpifield.kernels.spline_edges_pyccel import along_stencil, gauss_rule, path_cuts


def deposit_paths_spline_2d(
    start: "float[:,:]",
    end: "float[:,:]",
    charges: "float[:]",
    edges: "float[:,:,:]",
    lower_x: float,
    lower_y: float,
    cell_size_x: float,
    cell_size_y: float,
    num_x: int,
    num_y: int,
    periodic_x: int,
    periodic_y: int,
    degree: int,
) -> None:
    """Add the current of ``charges`` along the paths from ``start`` to ``end`` to ``edges``."""
    num_paths = start.shape[1]
    num = np.zeros(2, dtype=int)
    periodic = np.zeros(2, dtype=int)
    lower = np.zeros(2)
    inverse = np.zeros(2)
    num[0] = num_x
    periodic[0] = periodic_x
    lower[0] = lower_x
    inverse[0] = 1.0 / cell_size_x
    num[1] = num_y
    periodic[1] = periodic_y
    lower[1] = lower_y
    inverse[1] = 1.0 / cell_size_y
    r0 = np.zeros(2)
    r1 = np.zeros(2)
    a0 = np.zeros(2)
    a1 = np.zeros(2)
    p0 = np.zeros(2)
    step = np.zeros(2)
    cuts = np.zeros(32)
    point = np.zeros(5)
    weight = np.zeros(5)
    edge_x = np.zeros(4, dtype=int)
    along_x = np.zeros(4)
    exists_x = np.zeros(4, dtype=int)
    node_x = np.zeros(4, dtype=int)
    across_x = np.zeros(4)
    edge_y = np.zeros(4, dtype=int)
    along_y = np.zeros(4)
    exists_y = np.zeros(4, dtype=int)
    node_y = np.zeros(4, dtype=int)
    across_y = np.zeros(4)
    shift = 0.5 if degree % 2 == 0 else 0.0
    # the integrand of a piece has degree (p - 1) + (ndim - 1) p
    num_points = (degree - 1 + 1 * degree) // 2 + 1
    gauss_rule(num_points, point, weight)
    for path in range(num_paths):
        q = charges[path]
        if q == 0.0:
            continue
        finite = True
        for d in range(2):
            if not (start[d, path] - start[d, path] == 0.0):
                finite = False
            if not (end[d, path] - end[d, path] == 0.0):
                finite = False
        if not finite:
            continue
        extent = 0.0
        for d in range(2):
            r0[d] = (start[d, path] - lower[d]) * inverse[d]
            r1[d] = (end[d, path] - lower[d]) * inverse[d]
            span = r1[d] - r0[d] if r1[d] > r0[d] else r0[d] - r1[d]
            if span > extent:
                extent = span
        # stretches of at most 4 cells per axis keep the cuts within the buffer
        num_stretches = int(extent / 4.0) + 1
        for stretch in range(num_stretches):
            for d in range(2):
                a0[d] = r0[d] + (r1[d] - r0[d]) * stretch / num_stretches
                a1[d] = r0[d] + (r1[d] - r0[d]) * (stretch + 1) / num_stretches
            num_cuts = path_cuts(a0, a1, 2, shift, num, periodic, cuts)
            for piece in range(num_cuts - 1):
                t0 = cuts[piece]
                t1 = cuts[piece + 1]
                if t1 <= t0:
                    continue
                for d in range(2):
                    begin = a0[d] + t0 * (a1[d] - a0[d])
                    finish = a0[d] + t1 * (a1[d] - a0[d])
                    if periodic[d] == 0:
                        if begin < 0.0:
                            begin = 0.0
                        if begin > num[d] - 1:
                            begin = num[d] - 1.0
                        if finish < 0.0:
                            finish = 0.0
                        if finish > num[d] - 1:
                            finish = num[d] - 1.0
                    p0[d] = begin
                    step[d] = finish - begin
                for g in range(num_points):
                    rel_x = p0[0] + point[g] * step[0]
                    rel_y = p0[1] + point[g] * step[1]
                    amount = q * weight[g] * step[0]
                    if amount != 0.0:
                        along_stencil(
                            rel_x, num_x, periodic_x, degree, edge_x, along_x, exists_x
                        )
                        spline_cell(rel_y, num_y, periodic_y, degree, node_y, across_y)
                        for a in range(degree):
                            if exists_x[a] == 1:
                                for b in range(degree + 1):
                                    edges[edge_x[a], node_y[b], 0] += (
                                        amount * along_x[a] * across_y[b]
                                    )
                    amount = q * weight[g] * step[1]
                    if amount != 0.0:
                        spline_cell(rel_x, num_x, periodic_x, degree, node_x, across_x)
                        along_stencil(
                            rel_y, num_y, periodic_y, degree, edge_y, along_y, exists_y
                        )
                        for a in range(degree + 1):
                            for b in range(degree):
                                if exists_y[b] == 1:
                                    edges[node_x[a], edge_y[b], 1] += (
                                        amount * across_x[a] * along_y[b]
                                    )
