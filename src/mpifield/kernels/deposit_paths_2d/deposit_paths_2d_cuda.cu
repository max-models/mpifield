// The CUDA version (GPU) of the deposit_paths_2d kernel; deposit_paths_2d_pyccel.py next
// to this file is the Pyccel version (CPU), with the same arguments. Loaded by
// mpifield.kernels.catalog: one thread per path, atomic adds.
//
// The current of charges (N,) moving from start to end (2, N), added to edges (nx, ny, 2).

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "mpifield/kernels/linear_cell.cuh"
#include "mpifield/kernels/whitney_cell.cuh"

extern "C" __global__
void deposit_paths_2d(
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
    int periodic_y
) {
    const long long path = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (path >= charges.shape[0]) return;
    const double q = charges(path);
    if (q == 0.0) return;
    const double ax = (start(0, path) - lower_x) / cell_size_x;
    const double ay = (start(1, path) - lower_y) / cell_size_y;
    const double bx = (end(0, path) - lower_x) / cell_size_x;
    const double by = (end(1, path) - lower_y) / cell_size_y;
    if (!(ax > -1.0e300 && ax < 1.0e300 && ay > -1.0e300 && ay < 1.0e300
          && bx > -1.0e300 && bx < 1.0e300 && by > -1.0e300 && by < 1.0e300)) return;
    const double dx = bx - ax;
    const double dy = by - ay;
    double plane_x, step_x, plane_y, step_y;
    mpifield_first_plane(ax, dx, &plane_x, &step_x);
    mpifield_first_plane(ay, dy, &plane_y, &step_y);
    double time = 0.0;
    double x0 = mpifield_clamp_to_axis(ax, num_x, periodic_x);
    double y0 = mpifield_clamp_to_axis(ay, num_y, periodic_y);
    while (time < 1.0) {
        const double time_x = mpifield_plane_time(plane_x, ax, dx);
        const double time_y = mpifield_plane_time(plane_y, ay, dy);
        double following = 1.0;
        if (time_x < following) following = time_x;
        if (time_y < following) following = time_y;
        double x1, y1;
        if (following < 1.0) {
            x1 = mpifield_clamp_to_axis(ax + following * dx, num_x, periodic_x);
            y1 = mpifield_clamp_to_axis(ay + following * dy, num_y, periodic_y);
        } else {
            x1 = mpifield_clamp_to_axis(bx, num_x, periodic_x);
            y1 = mpifield_clamp_to_axis(by, num_y, periodic_y);
        }
        if (time_x <= following) plane_x += step_x;
        if (time_y <= following) plane_y += step_y;
        int i0, i1, j0, j1;
        double fx, fy;
        mpifield_linear_cell(0.5 * (x0 + x1), num_x, periodic_x, &i0, &i1, &fx);
        mpifield_linear_cell(0.5 * (y0 + y1), num_y, periodic_y, &j0, &j1, &fy);
        const double along_x = q * (x1 - x0);
        const double along_y = q * (y1 - y0);
        cunumpy_atomic_add(&edges(i0, j0, 0), mpifield_edge_weight(fy, 0) * along_x);
        cunumpy_atomic_add(&edges(i0, j1, 0), mpifield_edge_weight(fy, 1) * along_x);
        cunumpy_atomic_add(&edges(i0, j0, 1), mpifield_edge_weight(fx, 0) * along_y);
        cunumpy_atomic_add(&edges(i1, j0, 1), mpifield_edge_weight(fx, 1) * along_y);
        time = following;
        x0 = x1;
        y0 = y1;
    }
}
