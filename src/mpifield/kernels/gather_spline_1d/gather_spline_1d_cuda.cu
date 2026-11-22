// The CUDA version (GPU) of the gather_spline_1d kernel; gather_spline_1d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// B-spline interpolation of degree 0 to 3 of field (nx, components) at the
// markers into result (components, N); positions are (1, N).

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/spline_cell.cuh"

extern "C" __global__
void gather_spline_1d(
    Array2D<double> positions,
    Array2D<double> field,
    Array2D<double> result,
    double lower_x,
    double cell_size_x,
    int num_x,
    int periodic_x,
    int degree
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    int node_x[MPIFIELD_SPLINE_MAX_NODES];
    double weight_x[MPIFIELD_SPLINE_MAX_NODES];
    mpifield_spline_cell((positions(0, marker) - lower_x) / cell_size_x,
                         num_x, periodic_x, degree, node_x, weight_x);
    const int size = degree + 1;
    for (long long c = 0; c < field.shape[1]; ++c) {
        double total = 0.0;
        for (int a = 0; a < size; ++a)
            total += weight_x[a] * field(node_x[a], c);
        result(c, marker) = total;
    }
}
