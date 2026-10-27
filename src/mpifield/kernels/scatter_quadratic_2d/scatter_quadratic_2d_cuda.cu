// The CUDA version (GPU) of the scatter_quadratic_2d kernel; scatter_quadratic_2d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// Quadratic B-spline deposit of values (components, N) at positions (2, N) onto
// field (nx, ny, components), with atomic adds.

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "mpifield/kernels/quadratic_cell.cuh"

extern "C" __global__
void scatter_quadratic_2d(
    Array2D<double> positions,
    Array2D<double> values,
    Array3D<double> field,
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
    // Dead markers have zero weight. Do not read their positions, which may be
    // outside the domain (or non-finite), or issue zero atomics.
    bool deposit = false;
    for (long long c = 0; c < values.shape[0]; ++c) {
        if (values(c, marker) != 0.0) deposit = true;
    }
    if (!deposit) return;
    int node_x[3];
    double weight_x[3];
    int node_y[3];
    double weight_y[3];
    mpifield_quadratic_cell((positions(0, marker) - lower_x) / cell_size_x,
                            num_x, periodic_x, node_x, weight_x);
    mpifield_quadratic_cell((positions(1, marker) - lower_y) / cell_size_y,
                            num_y, periodic_y, node_y, weight_y);
    for (long long c = 0; c < values.shape[0]; ++c) {
        const double value = values(c, marker);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                cunumpy_atomic_add(&field(node_x[a], node_y[b], c),
                                   weight_x[a] * weight_y[b] * value);
    }
}
