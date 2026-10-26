// The CUDA version (GPU) of the gather_linear_2d kernel; gather_linear_2d_pyccel.py next to this
// file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// Bilinear interpolation of field (nx, ny, components) at the markers into
// result (components, N); positions are (2, N).

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/linear_cell.cuh"

extern "C" __global__
void gather_linear_2d(
    Array2D<double> positions,
    Array3D<double> field,
    Array2D<double> result,
    double lower_x,
    double lower_y,
    double cell_size_x,
    double cell_size_y,
    int num_x,
    int num_y,
    int periodic_x,
    int periodic_y
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    int i_left, i_right, j_left, j_right;
    double fx, fy;
    mpifield_linear_cell((positions(0, marker) - lower_x) / cell_size_x, num_x,
                       periodic_x, &i_left, &i_right, &fx);
    mpifield_linear_cell((positions(1, marker) - lower_y) / cell_size_y, num_y,
                       periodic_y, &j_left, &j_right, &fy);
    const double w_ll = (1.0 - fx) * (1.0 - fy);
    const double w_lr = fx * (1.0 - fy);
    const double w_ul = (1.0 - fx) * fy;
    const double w_ur = fx * fy;
    for (long long c = 0; c < field.shape[2]; ++c) {
        result(c, marker) = w_ll * field(i_left, j_left, c)
                          + w_lr * field(i_right, j_left, c)
                          + w_ul * field(i_left, j_right, c)
                          + w_ur * field(i_right, j_right, c);
    }
}
