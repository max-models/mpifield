// The CUDA version (GPU) of the gather_linear_3d kernel; gather_linear_3d_pyccel.py next to this
// file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// Trilinear interpolation at the markers (positions (3, N)) into result
// (components, N).
// field is the grid (nx, ny, nz, components) reshaped to (nx * ny * nz,
// components), since array views have at most three dimensions: cell (i, j, k)
// is row (i * ny + j) * nz + k.

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/linear_cell.cuh"

extern "C" __global__
void gather_linear_3d(
    Array2D<double> positions,
    Array2D<double> field,
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
    const double wx[2] = {1.0 - fx, fx};
    const double wy[2] = {1.0 - fy, fy};
    const double wz[2] = {1.0 - fz, fz};
    for (long long c = 0; c < field.shape[1]; ++c) {
        double value = 0.0;
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                for (int d = 0; d < 2; ++d)
                    value += wx[a] * wy[b] * wz[d]
                           * field(((long long)i[a] * num_y + j[b]) * num_z + k[d], c);
        result(c, marker) = value;
    }
}
