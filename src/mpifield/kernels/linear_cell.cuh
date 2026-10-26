// mpifield/kernels/linear_cell.cuh: the cell of a position along one grid axis.
//
// Shared by the CUDA versions of the deposit and gather kernels, and the same
// arithmetic as their Pyccel versions: on a periodic axis the `num` nodes
// cover `num` cells and the last node neighbours the first; on a bounded axis
// positions are clamped to the first and last cell.  The fraction (the weight
// of the right node) is clamped to [0, 1].

#ifndef MPIFIELD_KERNELS_LINEAR_CELL_CUH
#define MPIFIELD_KERNELS_LINEAR_CELL_CUH

__device__ __forceinline__ void mpifield_linear_cell(
    double rel, int num, int periodic, int* left, int* right, double* fraction
) {
    int l;
    if (periodic) {
        rel = rel - num * floor(rel / num);
        l = ((int)rel) % num;
        *right = (l + 1) % num;
    } else {
        l = rel >= 0.0 ? (int)rel : -1;
        if (l < 0) l = 0;
        if (l > num - 2) l = num - 2;
        *right = l + 1;
    }
    *left = l;
    double f = rel - l;
    if (f < 0.0) f = 0.0;
    if (f > 1.0) f = 1.0;
    *fraction = f;
}

#endif  // MPIFIELD_KERNELS_LINEAR_CELL_CUH
