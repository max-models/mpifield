// mpifield/kernels/spline_edges.cuh: the edge stencil, the path cuts and the Gauss
// rules of the spline edge kernels (gather_edges_spline_*, deposit_paths_spline_*).
//
// The same arithmetic as spline_edges_pyccel.py: the `degree` edges along an axis
// around a clamped position, with weights B_{p-1}(rel - k - 1/2) (bounded: edges
// 0 .. num - 2, one past them does not exist; periodic: wrapped), the sorted cut
// parameters of a path at knot planes and walls, and Gauss-Legendre rules on [0, 1].

#ifndef MPIFIELD_KERNELS_SPLINE_EDGES_CUH
#define MPIFIELD_KERNELS_SPLINE_EDGES_CUH

#include "mpifield/kernels/spline_cell.cuh"

#define MPIFIELD_SPLINE_MAX_CUTS 32

__device__ __forceinline__ int mpifield_floor_int(double x) {
    int whole = (int)x;
    if (x < whole) whole -= 1;
    return whole;
}

__device__ __forceinline__ void mpifield_along_stencil(
    double rel, int num, int periodic, int degree, int* edge, double* weight, int* exists
) {
    if (periodic == 2) {  // a clamped axis: the rescaled clamped B-splines
        mpifield_clamped_cell(rel, num, degree, 1, edge, weight);
        for (int a = 0; a < degree; ++a) exists[a] = 1;
        return;
    }
    const int lower = degree - 1;
    if (periodic == 1 && !(rel >= 0.0 && rel < num)) rel = rel - num * floor(rel / num);
    const double shifted = rel - 0.5;
    int middle;
    if (lower % 2 == 0) {
        middle = mpifield_floor_int(shifted + 0.5);
        if (periodic == 0 && middle > num - 2) middle = num - 2;
    } else {
        middle = mpifield_floor_int(shifted);
    }
    const double d = shifted - middle;
    if (lower == 0) {
        weight[0] = 1.0;
    } else if (lower == 1) {
        weight[0] = 1.0 - d;
        weight[1] = d;
    } else {
        weight[0] = 0.5 * (0.5 - d) * (0.5 - d);
        weight[1] = 0.75 - d * d;
        weight[2] = 0.5 * (0.5 + d) * (0.5 + d);
    }
    const int first = middle - lower / 2;
    for (int a = 0; a < degree; ++a) {
        const int k = first + a;
        if (periodic == 1) {
            edge[a] = (k + 2 * num) % num;
            exists[a] = 1;
        } else if (k < 0 || k > num - 2) {
            edge[a] = 0;
            exists[a] = 0;
        } else {
            edge[a] = k;
            exists[a] = 1;
        }
    }
}

__device__ __forceinline__ void mpifield_gauss_rule(int count, double* point, double* weight) {
    if (count == 1) {
        point[0] = 0.5; weight[0] = 1.0;
    } else if (count == 2) {
        point[0] = 0.21132486540518713; point[1] = 0.7886751345948129;
        weight[0] = 0.5; weight[1] = 0.5;
    } else if (count == 3) {
        point[0] = 0.1127016653792583; point[1] = 0.5; point[2] = 0.8872983346207417;
        weight[0] = 0.2777777777777778; weight[1] = 0.4444444444444444;
        weight[2] = 0.2777777777777778;
    } else if (count == 4) {
        point[0] = 0.06943184420297371; point[1] = 0.33000947820757187;
        point[2] = 0.6699905217924281; point[3] = 0.9305681557970262;
        weight[0] = 0.17392742256872692; weight[1] = 0.3260725774312731;
        weight[2] = 0.3260725774312731; weight[3] = 0.17392742256872692;
    } else {
        point[0] = 0.04691007703066802; point[1] = 0.23076534494715845; point[2] = 0.5;
        point[3] = 0.7692346550528415; point[4] = 0.9530899229693319;
        weight[0] = 0.11846344252809454; weight[1] = 0.2393143352496832;
        weight[2] = 0.28444444444444444; weight[3] = 0.2393143352496832;
        weight[4] = 0.11846344252809454;
    }
}

__device__ __forceinline__ int mpifield_path_cuts(
    const double* r0, const double* r1, int ndim, double shift, const int* num,
    const int* periodic, double* cuts
) {
    int count = 1;
    cuts[0] = 0.0;
    for (int d = 0; d < ndim; ++d) {
        const double delta = r1[d] - r0[d];
        if (delta == 0.0) continue;
        const double low = delta > 0.0 ? r0[d] : r1[d];
        const double high = delta > 0.0 ? r1[d] : r0[d];
        const double offset = periodic[d] == 2 ? 0.0 : shift;  // clamped knots: the nodes
        double plane = mpifield_floor_int(low - offset) + 1.0 + offset;
        while (plane < high) {
            const double t = (plane - r0[d]) / delta;
            if (t > 0.0 && t < 1.0) cuts[count++] = t;
            plane += 1.0;
        }
        if (periodic[d] != 1) {
            double t = (0.0 - r0[d]) / delta;
            if (t > 0.0 && t < 1.0) cuts[count++] = t;
            t = (num[d] - 1.0 - r0[d]) / delta;
            if (t > 0.0 && t < 1.0) cuts[count++] = t;
        }
    }
    cuts[count++] = 1.0;
    for (int i = 2; i < count - 1; ++i) {
        const double value = cuts[i];
        int j = i - 1;
        while (j > 0 && cuts[j] > value) {
            cuts[j + 1] = cuts[j];
            --j;
        }
        cuts[j + 1] = value;
    }
    return count;
}

// 1 where component `component` of a `form`-form is staggered along `axis`.
__device__ __forceinline__ int mpifield_staggered(int form, int ndim, int component, int axis) {
    if (form == ndim) return 1;
    if (form == 1) return axis == component ? 1 : 0;
    if (form == 2 && ndim == 3) return axis == component ? 0 : 1;
    return 0;
}

#endif  // MPIFIELD_KERNELS_SPLINE_EDGES_CUH
