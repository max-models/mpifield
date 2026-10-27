// The CUDA version (GPU) of the gather_quadratic_3d kernel; gather_quadratic_3d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// Quadratic B-spline interpolation of field (nx, ny, nz, components) at the markers into
// result (components, N); positions are (3, N).

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/quadratic_cell.cuh"

extern "C" __global__
void gather_quadratic_3d(
    Array2D<double> positions,
    Array4D<double> field,
    Array2D<double> result,
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
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    int node_x[3];
    double weight_x[3];
    int node_y[3];
    double weight_y[3];
    int node_z[3];
    double weight_z[3];
    mpifield_quadratic_cell((positions(0, marker) - lower_x) / cell_size_x,
                            num_x, periodic_x, node_x, weight_x);
    mpifield_quadratic_cell((positions(1, marker) - lower_y) / cell_size_y,
                            num_y, periodic_y, node_y, weight_y);
    mpifield_quadratic_cell((positions(2, marker) - lower_z) / cell_size_z,
                            num_z, periodic_z, node_z, weight_z);
    for (long long c = 0; c < field.shape[3]; ++c) {
        double total = 0.0;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int e = 0; e < 3; ++e)
                    total += weight_x[a] * weight_y[b] * weight_z[e]
                           * field(node_x[a], node_y[b], node_z[e], c);
        result(c, marker) = total;
    }
}
