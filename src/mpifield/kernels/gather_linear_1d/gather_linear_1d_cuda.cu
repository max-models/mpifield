// The CUDA version (GPU) of the gather_linear_1d kernel; gather_linear_1d_pyccel.py next to this
// file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// Linear interpolation of field (nx, components) at the markers into result
// (components, N); positions are (1, N).

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/linear_cell.cuh"

extern "C" __global__
void gather_linear_1d(
    Array2D<double> positions,
    Array2D<double> field,
    Array2D<double> result,
    double lower_x,
    double cell_size_x,
    int num_x,
    int periodic_x
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    int i_left, i_right;
    double fx;
    mpifield_linear_cell((positions(0, marker) - lower_x) / cell_size_x, num_x,
                       periodic_x, &i_left, &i_right, &fx);
    for (long long c = 0; c < field.shape[1]; ++c) {
        result(c, marker) = (1.0 - fx) * field(i_left, c) + fx * field(i_right, c);
    }
}
