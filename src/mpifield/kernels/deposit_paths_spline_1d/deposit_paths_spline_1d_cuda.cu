// The CUDA version (GPU) of the deposit_paths_spline_1d kernel; deposit_paths_spline_1d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per path, atomic adds.
//
// The current of charges (N,) moving from start to end (1, N) on the spline edge
// functions of degree 1 to 3, every piece between knot planes and walls integrated
// exactly by Gauss-Legendre quadrature.

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "mpifield/kernels/spline_cell.cuh"
#include "mpifield/kernels/spline_edges.cuh"

extern "C" __global__
void deposit_paths_spline_1d(
    Array2D<double> start,
    Array2D<double> end,
    Array1D<double> charges,
    Array2D<double> edges,
    double lower_x,
    double cell_size_x,
    int num_x,
    int periodic_x,
    int degree
) {
    const long long path = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (path >= start.shape[1]) return;
    const double q = charges(path);
    if (q == 0.0) return;
    const int num[1] = {num_x};
    const int periodic[1] = {periodic_x};
    const double lower[1] = {lower_x};
    const double cell[1] = {cell_size_x};
    double r0[1], r1[1], a0[1], a1[1], p0[1], step[1];
    double cuts[MPIFIELD_SPLINE_MAX_CUTS], point[5], weight[5];
    int edge_x[4]; double along_x[4]; int exists_x[4];
    double extent = 0.0;
    for (int d = 0; d < 1; ++d) {
        if (!(start(d, path) - start(d, path) == 0.0)) return;  // not finite
        if (!(end(d, path) - end(d, path) == 0.0)) return;
        r0[d] = (start(d, path) - lower[d]) / cell[d];
        r1[d] = (end(d, path) - lower[d]) / cell[d];
        const double span = r1[d] > r0[d] ? r1[d] - r0[d] : r0[d] - r1[d];
        if (span > extent) extent = span;
    }
    const double shift = degree % 2 == 0 ? 0.5 : 0.0;
    const int num_points = (degree - 1 + 0 * degree) / 2 + 1;
    mpifield_gauss_rule(num_points, point, weight);
    const int num_stretches = (int)(extent / 4.0) + 1;
    for (int stretch = 0; stretch < num_stretches; ++stretch) {
        for (int d = 0; d < 1; ++d) {
            a0[d] = r0[d] + (r1[d] - r0[d]) * stretch / num_stretches;
            a1[d] = r0[d] + (r1[d] - r0[d]) * (stretch + 1) / num_stretches;
        }
        const int num_cuts = mpifield_path_cuts(a0, a1, 1, shift, num, periodic, cuts);
        for (int piece = 0; piece < num_cuts - 1; ++piece) {
            const double t0 = cuts[piece], t1 = cuts[piece + 1];
            if (t1 <= t0) continue;
            for (int d = 0; d < 1; ++d) {
                double begin = a0[d] + t0 * (a1[d] - a0[d]);
                double finish = a0[d] + t1 * (a1[d] - a0[d]);
                if (!periodic[d]) {
                    if (begin < 0.0) begin = 0.0;
                    if (begin > num[d] - 1) begin = num[d] - 1;
                    if (finish < 0.0) finish = 0.0;
                    if (finish > num[d] - 1) finish = num[d] - 1;
                }
                p0[d] = begin;
                step[d] = finish - begin;
            }
            for (int g = 0; g < num_points; ++g) {
                const double rel_x = p0[0] + point[g] * step[0];
                {
                const double amount = q * weight[g] * step[0];
                if (amount != 0.0) {
                    mpifield_along_stencil(rel_x, num_x, periodic_x, degree, edge_x, along_x, exists_x);
                    for (int a = 0; a < degree; ++a)
                        if (exists_x[a])
                            cunumpy_atomic_add(&edges(edge_x[a], 0), amount * along_x[a]);
                }
                }
            }
        }
    }
}
