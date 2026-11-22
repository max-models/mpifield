// The CUDA version (GPU) of the gather_edges_spline_1d kernel; gather_edges_spline_1d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// The field of an edge field (*num, 1) at positions (1, N) into result (1, N),
// with the spline edge functions of degree 1 to 3.

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/spline_cell.cuh"
#include "mpifield/kernels/spline_edges.cuh"

extern "C" __global__
void gather_edges_spline_1d(
    Array2D<double> positions,
    Array2D<double> edges,
    Array2D<double> result,
    double lower_x,
    double cell_size_x,
    int num_x,
    int periodic_x,
    int degree
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    int edge_x[4]; double along_x[4]; int exists_x[4];
    double rel_x = (positions(0, marker) - lower_x) / cell_size_x;
    if (!periodic_x) { if (rel_x < 0.0) rel_x = 0.0; if (rel_x > num_x - 1) rel_x = num_x - 1; }
    mpifield_along_stencil(rel_x, num_x, periodic_x, degree, edge_x, along_x, exists_x);
    {
    double total = 0.0;
    for (int a = 0; a < degree; ++a)
        if (exists_x[a])
            total += along_x[a] * edges(edge_x[a], 0);
    result(0, marker) = total / cell_size_x;
    }
}
