"""The current of charges moving along straight paths, on the spline edge functions of degree 1 to 3, in 1D.

The Pyccel version (CPU) of the ``deposit_paths_spline_1d`` kernel; ``deposit_paths_spline_1d_cuda.cu``
next to this file is the CUDA version (GPU), with the same arguments. Each path, clamped
to the grid on bounded axes, is cut at the knot planes and walls (``spline_edges_pyccel``)
and every piece is integrated exactly by Gauss-Legendre quadrature:
``j_(k,a) += q * integral of W_(k,a) dr_a``. Serial (no OpenMP yet). A path with zero
charge or a non-finite end deposits nothing.

``start`` and ``end`` are ``(1, N)``, ``charges`` ``(N,)``, ``edges`` ``(*num, 1)``.
"""

import numpy as np

from mpifield.kernels.spline_edges_pyccel import along_stencil, gauss_rule, path_cuts


def deposit_paths_spline_1d(
    start: "float[:,:]",
    end: "float[:,:]",
    charges: "float[:]",
    edges: "float[:,:]",
    lower_x: float,
    cell_size_x: float,
    num_x: int,
    periodic_x: int,
    degree: int,
) -> None:
    """Add the current of ``charges`` along the paths from ``start`` to ``end`` to ``edges``."""
    num_paths = start.shape[1]
    num = np.zeros(1, dtype=int)
    periodic = np.zeros(1, dtype=int)
    lower = np.zeros(1)
    inverse = np.zeros(1)
    num[0] = num_x
    periodic[0] = periodic_x
    lower[0] = lower_x
    inverse[0] = 1.0 / cell_size_x
    r0 = np.zeros(1)
    r1 = np.zeros(1)
    a0 = np.zeros(1)
    a1 = np.zeros(1)
    p0 = np.zeros(1)
    step = np.zeros(1)
    cuts = np.zeros(32)
    point = np.zeros(5)
    weight = np.zeros(5)
    edge_x = np.zeros(4, dtype=int)
    along_x = np.zeros(4)
    exists_x = np.zeros(4, dtype=int)
    shift = 0.5 if degree % 2 == 0 else 0.0
    # the integrand of a piece has degree (p - 1) + (ndim - 1) p
    num_points = (degree - 1 + 0 * degree) // 2 + 1
    gauss_rule(num_points, point, weight)
    for path in range(num_paths):
        q = charges[path]
        if q == 0.0:
            continue
        finite = True
        for d in range(1):
            if not (start[d, path] - start[d, path] == 0.0):
                finite = False
            if not (end[d, path] - end[d, path] == 0.0):
                finite = False
        if not finite:
            continue
        extent = 0.0
        for d in range(1):
            r0[d] = (start[d, path] - lower[d]) * inverse[d]
            r1[d] = (end[d, path] - lower[d]) * inverse[d]
            span = r1[d] - r0[d] if r1[d] > r0[d] else r0[d] - r1[d]
            if span > extent:
                extent = span
        # stretches of at most 4 cells per axis keep the cuts within the buffer
        num_stretches = int(extent / 4.0) + 1
        for stretch in range(num_stretches):
            for d in range(1):
                a0[d] = r0[d] + (r1[d] - r0[d]) * stretch / num_stretches
                a1[d] = r0[d] + (r1[d] - r0[d]) * (stretch + 1) / num_stretches
            num_cuts = path_cuts(a0, a1, 1, shift, num, periodic, cuts)
            for piece in range(num_cuts - 1):
                t0 = cuts[piece]
                t1 = cuts[piece + 1]
                if t1 <= t0:
                    continue
                for d in range(1):
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
                    amount = q * weight[g] * step[0]
                    if amount != 0.0:
                        along_stencil(
                            rel_x, num_x, periodic_x, degree, edge_x, along_x, exists_x
                        )
                        for a in range(degree):
                            if exists_x[a] == 1:
                                edges[edge_x[a], 0] += amount * along_x[a]
