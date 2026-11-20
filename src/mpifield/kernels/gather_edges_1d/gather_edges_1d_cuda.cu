// The CUDA version (GPU) of the gather_edges_1d kernel; gather_edges_1d_pyccel.py next to
// this file is the Pyccel version (CPU), with the same arguments. Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// The edge field of edges (nx, 1) at positions (1, N) into result (1, N).

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/linear_cell.cuh"

extern "C" __global__
void gather_edges_1d(
    Array2D<double> positions,
    Array2D<double> edges,
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
    result(0, marker) = edges(i_left, 0) / cell_size_x;
}
