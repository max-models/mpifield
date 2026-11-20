// The CUDA version (GPU) of the gather_edges_3d kernel; gather_edges_3d_pyccel.py next to
// this file is the Pyccel version (CPU), with the same arguments. Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// The edge field of edges (nx, ny, nz, 3) at positions (3, N) into result (3, N).

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/linear_cell.cuh"
#include "mpifield/kernels/whitney_cell.cuh"

extern "C" __global__
void gather_edges_3d(
    Array2D<double> positions,
    Array4D<double> edges,
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
    int i[2], j[2], k[2];
    double fx, fy, fz;
    mpifield_linear_cell((positions(0, marker) - lower_x) / cell_size_x, num_x,
                         periodic_x, &i[0], &i[1], &fx);
    mpifield_linear_cell((positions(1, marker) - lower_y) / cell_size_y, num_y,
                         periodic_y, &j[0], &j[1], &fy);
    mpifield_linear_cell((positions(2, marker) - lower_z) / cell_size_z, num_z,
                         periodic_z, &k[0], &k[1], &fz);
    double ex = 0.0, ey = 0.0, ez = 0.0;
    for (int u = 0; u < 2; ++u) {
        for (int w = 0; w < 2; ++w) {
            ex += mpifield_edge_weight(fy, u) * mpifield_edge_weight(fz, w) * edges(i[0], j[u], k[w], 0);
            ey += mpifield_edge_weight(fx, u) * mpifield_edge_weight(fz, w) * edges(i[u], j[0], k[w], 1);
            ez += mpifield_edge_weight(fx, u) * mpifield_edge_weight(fy, w) * edges(i[u], j[w], k[0], 2);
        }
    }
    result(0, marker) = ex / cell_size_x;
    result(1, marker) = ey / cell_size_y;
    result(2, marker) = ez / cell_size_z;
}
