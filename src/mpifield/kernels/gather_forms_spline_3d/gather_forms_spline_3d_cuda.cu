// The CUDA version (GPU) of the gather_forms_spline_3d kernel; gather_forms_spline_3d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// The gather of a k-form of B-spline degree 1 to 3 (spline_forms.py): B_p along the
// axes a component is not staggered on, B_(p-1) shifted by half a cell along the others.

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/spline_cell.cuh"
#include "mpifield/kernels/spline_edges.cuh"

extern "C" __global__
void gather_forms_spline_3d(
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
    int periodic_z,
    int degree,
    int form
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    double rel_x = (positions(0, marker) - lower_x) / cell_size_x;
    if (periodic_x != 1) { if (rel_x < 0.0) rel_x = 0.0; if (rel_x > num_x - 1) rel_x = num_x - 1; }
    int index_x[4]; double weight_x[4]; int exists_x[4]; int size_x;
    double rel_y = (positions(1, marker) - lower_y) / cell_size_y;
    if (periodic_y != 1) { if (rel_y < 0.0) rel_y = 0.0; if (rel_y > num_y - 1) rel_y = num_y - 1; }
    int index_y[4]; double weight_y[4]; int exists_y[4]; int size_y;
    double rel_z = (positions(2, marker) - lower_z) / cell_size_z;
    if (periodic_z != 1) { if (rel_z < 0.0) rel_z = 0.0; if (rel_z > num_z - 1) rel_z = num_z - 1; }
    int index_z[4]; double weight_z[4]; int exists_z[4]; int size_z;
    for (long long c = 0; c < field.shape[3]; ++c) {
        if (mpifield_staggered(form, 3, (int)c, 0)) {
            mpifield_along_stencil(rel_x, num_x, periodic_x, degree, index_x, weight_x, exists_x);
            size_x = degree;
        } else {
            mpifield_spline_cell(rel_x, num_x, periodic_x, degree, index_x, weight_x);
            for (int i = 0; i <= degree; ++i) exists_x[i] = 1;
            size_x = degree + 1;
        }
        if (mpifield_staggered(form, 3, (int)c, 1)) {
            mpifield_along_stencil(rel_y, num_y, periodic_y, degree, index_y, weight_y, exists_y);
            size_y = degree;
        } else {
            mpifield_spline_cell(rel_y, num_y, periodic_y, degree, index_y, weight_y);
            for (int i = 0; i <= degree; ++i) exists_y[i] = 1;
            size_y = degree + 1;
        }
        if (mpifield_staggered(form, 3, (int)c, 2)) {
            mpifield_along_stencil(rel_z, num_z, periodic_z, degree, index_z, weight_z, exists_z);
            size_z = degree;
        } else {
            mpifield_spline_cell(rel_z, num_z, periodic_z, degree, index_z, weight_z);
            for (int i = 0; i <= degree; ++i) exists_z[i] = 1;
            size_z = degree + 1;
        }
        double total = 0.0;
        for (int a = 0; a < size_x; ++a)
            if (exists_x[a])
                for (int b = 0; b < size_y; ++b)
                    if (exists_y[b])
                        for (int e = 0; e < size_z; ++e)
                            if (exists_z[e])
                                total += weight_x[a] * weight_y[b] * weight_z[e] * field(index_x[a], index_y[b], index_z[e], c);
        result(c, marker) = total;
    }
}
