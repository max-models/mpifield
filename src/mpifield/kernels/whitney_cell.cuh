// mpifield/kernels/whitney_cell.cuh: the weights of the lowest-order edge (Whitney)
// functions of a cell, and the steps of a path through the grid.
//
// The CUDA version of whitney_cell_pyccel.py (Pyccel) and whitney_cell.py (NumPy), with
// the same arithmetic. Shared by the CUDA versions of the edge gather and path deposit
// kernels, and by other packages' kernels (picnic's implicit movers).

#ifndef MPIFIELD_KERNELS_WHITNEY_CELL_CUH
#define MPIFIELD_KERNELS_WHITNEY_CELL_CUH

// The linear weight of the lower (upper 0) or upper (1) edge at fraction.
__device__ __forceinline__ double mpifield_edge_weight(double fraction, int upper) {
    return upper ? fraction : 1.0 - fraction;
}

// The average of a 3D edge's bilinear weight over a straight piece whose midpoint is at
// fraction_u, fraction_w across the cell and whose lengths along the two other axes are
// delta_u, delta_w (in cells): the midpoint's weight plus or minus delta_u delta_w / 12.
__device__ __forceinline__ double mpifield_edge_path_weight(
    double fraction_u, double fraction_w, double delta_u, double delta_w,
    int upper_u, int upper_w
) {
    const double cross = delta_u * delta_w / 12.0;
    return mpifield_edge_weight(fraction_u, upper_u) * mpifield_edge_weight(fraction_w, upper_w)
           + (upper_u == upper_w ? cross : -cross);
}

// rel on a bounded axis clamped to [0, num - 1]; unchanged if periodic.
__device__ __forceinline__ double mpifield_clamp_to_axis(double rel, int num, int periodic) {
    if (periodic) return rel;
    if (rel < 0.0) return 0.0;
    if (rel > num - 1.0) return num - 1.0;
    return rel;
}

// The time at which rel + t delta reaches plane; 2 (never within the path) for delta 0.
__device__ __forceinline__ double mpifield_plane_time(double plane, double rel, double delta) {
    return delta != 0.0 ? (plane - rel) / delta : 2.0;
}

// The first grid line a coordinate at rel moving by delta reaches, and the step to the
// one after it.
__device__ __forceinline__ void mpifield_first_plane(
    double rel, double delta, double* plane, double* step
) {
    if (delta < 0.0) {
        *plane = ceil(rel) - 1.0;
        *step = -1.0;
    } else {
        *plane = floor(rel) + 1.0;
        *step = 1.0;
    }
}

#endif  // MPIFIELD_KERNELS_WHITNEY_CELL_CUH
