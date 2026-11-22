// The CUDA version (GPU) of the scatter_spline_3d kernel; scatter_spline_3d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// B-spline deposit of degree 0 to 3 of values (components, N) at positions (3, N)
// onto field (nx, ny, nz, components), with atomic adds.

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "mpifield/kernels/spline_cell.cuh"

extern "C" __global__
void scatter_spline_3d(
    Array2D<double> positions,
    Array2D<double> values,
    Array4D<double> field,
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
    int periodic_z,
    int degree
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
    int node_x[MPIFIELD_SPLINE_MAX_NODES];
    double weight_x[MPIFIELD_SPLINE_MAX_NODES];
    int node_y[MPIFIELD_SPLINE_MAX_NODES];
    double weight_y[MPIFIELD_SPLINE_MAX_NODES];
    int node_z[MPIFIELD_SPLINE_MAX_NODES];
    double weight_z[MPIFIELD_SPLINE_MAX_NODES];
    mpifield_spline_cell((positions(0, marker) - lower_x) / cell_size_x,
                         num_x, periodic_x, degree, node_x, weight_x);
    mpifield_spline_cell((positions(1, marker) - lower_y) / cell_size_y,
                         num_y, periodic_y, degree, node_y, weight_y);
    mpifield_spline_cell((positions(2, marker) - lower_z) / cell_size_z,
                         num_z, periodic_z, degree, node_z, weight_z);
    const int size = degree + 1;
    for (long long c = 0; c < values.shape[0]; ++c) {
        const double value = values(c, marker);
        for (int a = 0; a < size; ++a)
            for (int b = 0; b < size; ++b)
                for (int e = 0; e < size; ++e)
                    cunumpy_atomic_add(&field(node_x[a], node_y[b], node_z[e], c),
                                       weight_x[a] * weight_y[b] * weight_z[e] * value);
    }
}
