// The CUDA version (GPU) of the gather_edges_spline_3d kernel; gather_edges_spline_3d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// The field of an edge field (*num, 3) at positions (3, N) into result (3, N),
// with the spline edge functions of degree 1 to 3.

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/spline_cell.cuh"
#include "mpifield/kernels/spline_edges.cuh"

extern "C" __global__
void gather_edges_spline_3d(
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
    int periodic_z,
    int degree
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    int edge_x[4]; double along_x[4]; int exists_x[4];
    int node_x[4]; double across_x[4];
    int edge_y[4]; double along_y[4]; int exists_y[4];
    int node_y[4]; double across_y[4];
    int edge_z[4]; double along_z[4]; int exists_z[4];
    int node_z[4]; double across_z[4];
    double rel_x = (positions(0, marker) - lower_x) / cell_size_x;
    if (!periodic_x) { if (rel_x < 0.0) rel_x = 0.0; if (rel_x > num_x - 1) rel_x = num_x - 1; }
    double rel_y = (positions(1, marker) - lower_y) / cell_size_y;
    if (!periodic_y) { if (rel_y < 0.0) rel_y = 0.0; if (rel_y > num_y - 1) rel_y = num_y - 1; }
    double rel_z = (positions(2, marker) - lower_z) / cell_size_z;
    if (!periodic_z) { if (rel_z < 0.0) rel_z = 0.0; if (rel_z > num_z - 1) rel_z = num_z - 1; }
    mpifield_along_stencil(rel_x, num_x, periodic_x, degree, edge_x, along_x, exists_x);
    mpifield_spline_cell(rel_y, num_y, periodic_y, degree, node_y, across_y);
    mpifield_spline_cell(rel_z, num_z, periodic_z, degree, node_z, across_z);
    {
    double total = 0.0;
    for (int a = 0; a < degree; ++a)
        if (exists_x[a])
            for (int b = 0; b < degree + 1; ++b)
                for (int e = 0; e < degree + 1; ++e)
                    total += along_x[a] * across_y[b] * across_z[e] * edges(edge_x[a], node_y[b], node_z[e], 0);
    result(0, marker) = total / cell_size_x;
    }
    mpifield_spline_cell(rel_x, num_x, periodic_x, degree, node_x, across_x);
    mpifield_along_stencil(rel_y, num_y, periodic_y, degree, edge_y, along_y, exists_y);
    mpifield_spline_cell(rel_z, num_z, periodic_z, degree, node_z, across_z);
    {
    double total = 0.0;
    for (int a = 0; a < degree + 1; ++a)
        for (int b = 0; b < degree; ++b)
            if (exists_y[b])
                for (int e = 0; e < degree + 1; ++e)
                    total += across_x[a] * along_y[b] * across_z[e] * edges(node_x[a], edge_y[b], node_z[e], 1);
    result(1, marker) = total / cell_size_y;
    }
    mpifield_spline_cell(rel_x, num_x, periodic_x, degree, node_x, across_x);
    mpifield_spline_cell(rel_y, num_y, periodic_y, degree, node_y, across_y);
    mpifield_along_stencil(rel_z, num_z, periodic_z, degree, edge_z, along_z, exists_z);
    {
    double total = 0.0;
    for (int a = 0; a < degree + 1; ++a)
        for (int b = 0; b < degree + 1; ++b)
            for (int e = 0; e < degree; ++e)
                if (exists_z[e])
                    total += across_x[a] * across_y[b] * along_z[e] * edges(node_x[a], node_y[b], edge_z[e], 2);
    result(2, marker) = total / cell_size_z;
    }
}
