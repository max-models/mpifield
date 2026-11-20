// The CUDA version (GPU) of the gather_edges_2d kernel; gather_edges_2d_pyccel.py next to
// this file is the Pyccel version (CPU), with the same arguments. Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// The edge field of edges (nx, ny, 2) at positions (2, N) into result (2, N).

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/linear_cell.cuh"
#include "mpifield/kernels/whitney_cell.cuh"

extern "C" __global__
void gather_edges_2d(
    Array2D<double> positions,
    Array3D<double> edges,
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
    int i0, i1, j0, j1;
    double fx, fy;
    mpifield_linear_cell((positions(0, marker) - lower_x) / cell_size_x, num_x,
                         periodic_x, &i0, &i1, &fx);
    mpifield_linear_cell((positions(1, marker) - lower_y) / cell_size_y, num_y,
                         periodic_y, &j0, &j1, &fy);
    result(0, marker) = (mpifield_edge_weight(fy, 0) * edges(i0, j0, 0)
                         + mpifield_edge_weight(fy, 1) * edges(i0, j1, 0)) / cell_size_x;
    result(1, marker) = (mpifield_edge_weight(fx, 0) * edges(i0, j0, 1)
                         + mpifield_edge_weight(fx, 1) * edges(i1, j0, 1)) / cell_size_y;
}
