// The CUDA version (GPU) of the gather_forms_spline_1d kernel; gather_forms_spline_1d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: one thread per marker.
//
// The gather of a k-form of B-spline degree 1 to 3 (spline_forms.py): B_p along the
// axes a component is not staggered on, B_(p-1) shifted by half a cell along the others.

#include "cunumpy/array_view.cuh"
#include "mpifield/kernels/spline_cell.cuh"
#include "mpifield/kernels/spline_edges.cuh"

extern "C" __global__
void gather_forms_spline_1d(
    Array2D<double> positions,
    Array2D<double> field,
    Array2D<double> result,
    double lower_x,
    double cell_size_x,
    int num_x,
    int periodic_x,
    int degree,
    int form
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    double rel_x = (positions(0, marker) - lower_x) / cell_size_x;
    if (periodic_x != 1) { if (rel_x < 0.0) rel_x = 0.0; if (rel_x > num_x - 1) rel_x = num_x - 1; }
    int index_x[4]; double weight_x[4]; int exists_x[4]; int size_x;
    for (long long c = 0; c < field.shape[1]; ++c) {
        if (mpifield_staggered(form, 1, (int)c, 0)) {
            mpifield_along_stencil(rel_x, num_x, periodic_x, degree, index_x, weight_x, exists_x);
            size_x = degree;
        } else {
            mpifield_spline_cell(rel_x, num_x, periodic_x, degree, index_x, weight_x);
            for (int i = 0; i <= degree; ++i) exists_x[i] = 1;
            size_x = degree + 1;
        }
        double total = 0.0;
        for (int a = 0; a < size_x; ++a)
            if (exists_x[a])
                total += weight_x[a] * field(index_x[a], c);
        result(c, marker) = total;
    }
}
