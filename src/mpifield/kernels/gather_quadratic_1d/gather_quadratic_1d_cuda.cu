// The CUDA version (GPU) of the gather_quadratic_1d kernel; gather_quadratic_1d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// Quadratic B-spline interpolation of field (nx, components) at the markers into
// result (components, N); positions are (1, N).

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/quadratic_cell.cuh"

extern "C" __global__
void gather_quadratic_1d(
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
    int node_x[3];
    double weight_x[3];
    mpifield_quadratic_cell((positions(0, marker) - lower_x) / cell_size_x,
                            num_x, periodic_x, node_x, weight_x);
    for (long long c = 0; c < field.shape[1]; ++c) {
        double total = 0.0;
        for (int a = 0; a < 3; ++a) total += weight_x[a] * field(node_x[a], c);
        result(c, marker) = total;
    }
}
