// The CUDA version (GPU) of the deposit_paths_3d kernel; deposit_paths_3d_pyccel.py next
// to this file is the Pyccel version (CPU), with the same arguments. Loaded by
// mpifield.kernels.catalog: one thread per path, atomic adds.
//
// The current of charges (N,) moving from start to end (3, N), added to edges
// (nx, ny, nz, 3).

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "mpifield/kernels/linear_cell.cuh"
#include "mpifield/kernels/whitney_cell.cuh"

extern "C" __global__
void deposit_paths_3d(
    Array2D<double> start,
    Array2D<double> end,
    Array1D<double> charges,
    Array4D<double> edges,
    double lower_x,
    double lower_y,
    double lower_z,
    double cell_size_x,
    double cell_size_y,
    double cell_size_z,
    int num_x,
    int num_y,
    int num_z,
    int periodic_x,
    int periodic_y,
    int periodic_z
) {
    const long long path = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (path >= charges.shape[0]) return;
    const double q = charges(path);
    if (q == 0.0) return;
    const double lower[3] = {lower_x, lower_y, lower_z};
    const double size[3] = {cell_size_x, cell_size_y, cell_size_z};
    const int num[3] = {num_x, num_y, num_z};
    const int periodic[3] = {periodic_x, periodic_y, periodic_z};
    double a[3], b[3], d[3], plane[3], step[3], p0[3], p1[3];
    for (int axis = 0; axis < 3; ++axis) {
        a[axis] = (start(axis, path) - lower[axis]) / size[axis];
        b[axis] = (end(axis, path) - lower[axis]) / size[axis];
        if (!(a[axis] > -1.0e300 && a[axis] < 1.0e300 && b[axis] > -1.0e300
              && b[axis] < 1.0e300)) return;
    }
    for (int axis = 0; axis < 3; ++axis) {
        d[axis] = b[axis] - a[axis];
        mpifield_first_plane(a[axis], d[axis], &plane[axis], &step[axis]);
        p0[axis] = mpifield_clamp_to_axis(a[axis], num[axis], periodic[axis]);
    }
    double time = 0.0;
    while (time < 1.0) {
        double times[3];
        double following = 1.0;
        for (int axis = 0; axis < 3; ++axis) {
            times[axis] = mpifield_plane_time(plane[axis], a[axis], d[axis]);
            if (times[axis] < following) following = times[axis];
        }
        for (int axis = 0; axis < 3; ++axis) {
            p1[axis] = mpifield_clamp_to_axis(
                following < 1.0 ? a[axis] + following * d[axis] : b[axis],
                num[axis], periodic[axis]);
            if (times[axis] <= following) plane[axis] += step[axis];
        }
        int lo[3], hi[3];
        double f[3], delta[3];
        for (int axis = 0; axis < 3; ++axis) {
            mpifield_linear_cell(0.5 * (p0[axis] + p1[axis]), num[axis], periodic[axis],
                                 &lo[axis], &hi[axis], &f[axis]);
            delta[axis] = p1[axis] - p0[axis];
        }
        for (int u = 0; u < 2; ++u) {
            for (int w = 0; w < 2; ++w) {
                cunumpy_atomic_add(&edges(lo[0], u ? hi[1] : lo[1], w ? hi[2] : lo[2], 0),
                    q * delta[0] * mpifield_edge_path_weight(f[1], f[2], delta[1], delta[2], u, w));
                cunumpy_atomic_add(&edges(u ? hi[0] : lo[0], lo[1], w ? hi[2] : lo[2], 1),
                    q * delta[1] * mpifield_edge_path_weight(f[0], f[2], delta[0], delta[2], u, w));
                cunumpy_atomic_add(&edges(u ? hi[0] : lo[0], w ? hi[1] : lo[1], lo[2], 2),
                    q * delta[2] * mpifield_edge_path_weight(f[0], f[1], delta[0], delta[1], u, w));
            }
        }
        time = following;
        for (int axis = 0; axis < 3; ++axis) p0[axis] = p1[axis];
    }
}
