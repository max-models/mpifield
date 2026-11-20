// The CUDA version (GPU) of the deposit_paths_1d kernel; deposit_paths_1d_pyccel.py next
// to this file is the Pyccel version (CPU), with the same arguments. Loaded by
// mpifield.kernels.catalog: one thread per path, atomic adds.
//
// The current of charges (N,) moving from start to end (1, N), added to edges (nx, 1).

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "mpifield/kernels/linear_cell.cuh"
#include "mpifield/kernels/whitney_cell.cuh"

extern "C" __global__
void deposit_paths_1d(
    Array2D<double> start,
    Array2D<double> end,
    Array1D<double> charges,
    Array2D<double> edges,
    double lower_x,
    double cell_size_x,
    int num_x,
    int periodic_x
) {
    const long long path = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (path >= charges.shape[0]) return;
    const double q = charges(path);
    if (q == 0.0) return;
    const double ax = (start(0, path) - lower_x) / cell_size_x;
    const double bx = (end(0, path) - lower_x) / cell_size_x;
    if (!(ax > -1.0e300 && ax < 1.0e300 && bx > -1.0e300 && bx < 1.0e300)) return;
    const double dx = bx - ax;
    double plane_x, step_x;
    mpifield_first_plane(ax, dx, &plane_x, &step_x);
    double time = 0.0;
    double x0 = mpifield_clamp_to_axis(ax, num_x, periodic_x);
    while (time < 1.0) {
        double following = mpifield_plane_time(plane_x, ax, dx);
        double x1;
        if (following < 1.0) {
            x1 = mpifield_clamp_to_axis(ax + following * dx, num_x, periodic_x);
            plane_x += step_x;
        } else {
            following = 1.0;
            x1 = mpifield_clamp_to_axis(bx, num_x, periodic_x);
        }
        int i0, i1;
        double fx;
        mpifield_linear_cell(0.5 * (x0 + x1), num_x, periodic_x, &i0, &i1, &fx);
        cunumpy_atomic_add(&edges(i0, 0), q * (x1 - x0));
        time = following;
        x0 = x1;
    }
}
