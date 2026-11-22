// The CUDA version (GPU) of the deposit_paths_spline_2d kernel; deposit_paths_spline_2d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per path, atomic adds.
//
// The current of charges (N,) moving from start to end (2, N) on the spline edge
// functions of degree 1 to 3, every piece between knot planes and walls integrated
// exactly by Gauss-Legendre quadrature.

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "mpifield/kernels/spline_cell.cuh"
#include "mpifield/kernels/spline_edges.cuh"

extern "C" __global__
void deposit_paths_spline_2d(
    Array2D<double> start,
    Array2D<double> end,
    Array1D<double> charges,
    Array3D<double> edges,
    double lower_x,
    double lower_y,
    double cell_size_x,
    double cell_size_y,
    int num_x,
    int num_y,
    int periodic_x,
    int periodic_y,
    int degree
) {
    const long long path = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (path >= start.shape[1]) return;
    const double q = charges(path);
    if (q == 0.0) return;
    const int num[2] = {num_x, num_y};
    const int periodic[2] = {periodic_x, periodic_y};
    const double lower[2] = {lower_x, lower_y};
    const double cell[2] = {cell_size_x, cell_size_y};
    double r0[2], r1[2], a0[2], a1[2], p0[2], step[2];
    double cuts[MPIFIELD_SPLINE_MAX_CUTS], point[5], weight[5];
    int edge_x[4]; double along_x[4]; int exists_x[4];
    int node_x[4]; double across_x[4];
    int edge_y[4]; double along_y[4]; int exists_y[4];
    int node_y[4]; double across_y[4];
    double extent = 0.0;
    for (int d = 0; d < 2; ++d) {
        if (!(start(d, path) - start(d, path) == 0.0)) return;  // not finite
        if (!(end(d, path) - end(d, path) == 0.0)) return;
        r0[d] = (start(d, path) - lower[d]) / cell[d];
        r1[d] = (end(d, path) - lower[d]) / cell[d];
        const double span = r1[d] > r0[d] ? r1[d] - r0[d] : r0[d] - r1[d];
        if (span > extent) extent = span;
    }
    const double shift = degree % 2 == 0 ? 0.5 : 0.0;
    const int num_points = (degree - 1 + 1 * degree) / 2 + 1;
    mpifield_gauss_rule(num_points, point, weight);
    const int num_stretches = (int)(extent / 4.0) + 1;
    for (int stretch = 0; stretch < num_stretches; ++stretch) {
        for (int d = 0; d < 2; ++d) {
            a0[d] = r0[d] + (r1[d] - r0[d]) * stretch / num_stretches;
            a1[d] = r0[d] + (r1[d] - r0[d]) * (stretch + 1) / num_stretches;
        }
        const int num_cuts = mpifield_path_cuts(a0, a1, 2, shift, num, periodic, cuts);
        for (int piece = 0; piece < num_cuts - 1; ++piece) {
            const double t0 = cuts[piece], t1 = cuts[piece + 1];
            if (t1 <= t0) continue;
            for (int d = 0; d < 2; ++d) {
                double begin = a0[d] + t0 * (a1[d] - a0[d]);
                double finish = a0[d] + t1 * (a1[d] - a0[d]);
                if (periodic[d] != 1) {
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
                const double rel_y = p0[1] + point[g] * step[1];
                {
                const double amount = q * weight[g] * step[0];
                if (amount != 0.0) {
                    mpifield_along_stencil(rel_x, num_x, periodic_x, degree, edge_x, along_x, exists_x);
                    mpifield_spline_cell(rel_y, num_y, periodic_y, degree, node_y, across_y);
                    for (int a = 0; a < degree; ++a)
                        if (exists_x[a])
                            for (int b = 0; b < degree + 1; ++b)
                                cunumpy_atomic_add(&edges(edge_x[a], node_y[b], 0), amount * along_x[a] * across_y[b]);
                }
                }
                {
                const double amount = q * weight[g] * step[1];
                if (amount != 0.0) {
                    mpifield_spline_cell(rel_x, num_x, periodic_x, degree, node_x, across_x);
                    mpifield_along_stencil(rel_y, num_y, periodic_y, degree, edge_y, along_y, exists_y);
                    for (int a = 0; a < degree + 1; ++a)
                        for (int b = 0; b < degree; ++b)
                            if (exists_y[b])
                                cunumpy_atomic_add(&edges(node_x[a], edge_y[b], 1), amount * across_x[a] * along_y[b]);
                }
                }
            }
        }
    }
}
