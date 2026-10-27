// The CUDA version (GPU) of the gather_quadratic_2d kernel; gather_quadratic_2d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// Quadratic B-spline interpolation of field (nx, ny, components) at the markers into
// result (components, N); positions are (2, N).

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/quadratic_cell.cuh"

extern "C" __global__
void gather_quadratic_2d(
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
    int node_x[3];
    double weight_x[3];
    int node_y[3];
    double weight_y[3];
    mpifield_quadratic_cell((positions(0, marker) - lower_x) / cell_size_x,
                            num_x, periodic_x, node_x, weight_x);
    mpifield_quadratic_cell((positions(1, marker) - lower_y) / cell_size_y,
                            num_y, periodic_y, node_y, weight_y);
    for (long long c = 0; c < field.shape[2]; ++c) {
        double total = 0.0;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                total += weight_x[a] * weight_y[b] * field(node_x[a], node_y[b], c);
        result(c, marker) = total;
    }
}
