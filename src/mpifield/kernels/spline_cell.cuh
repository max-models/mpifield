// mpifield/kernels/spline_cell.cuh: the stencil of a position along one grid
// axis for B-spline weights of degree 0 to 3.
//
// Shared by the CUDA versions of the spline deposit and gather kernels, and the
// same arithmetic as their Pyccel versions (spline_cell_pyccel.py): the stencil of
// an even degree is centred on the nearest node, the one of an odd degree on the
// cell holding the position.  On a periodic axis the `num` nodes cover `num`
// cells and a position wraps into the period; on a bounded axis it is clamped
// to [0, num - 1] and a stencil node past the edge is the edge node.

#ifndef MPIFIELD_KERNELS_SPLINE_CELL_CUH
#define MPIFIELD_KERNELS_SPLINE_CELL_CUH

// The most nodes of a stencil along one axis (degree 3).
#define MPIFIELD_SPLINE_MAX_NODES 4

__device__ __forceinline__ void mpifield_spline_cell(
    double rel, int num, int periodic, int degree, int* node, double* weight
) {
    if (periodic) {
        if (!(rel >= 0.0 && rel < num)) rel = rel - num * floor(rel / num);
    } else {
        if (rel < 0.0) rel = 0.0;
        if (rel > num - 1) rel = num - 1;
    }
    const int middle = degree % 2 == 0 ? (int)(rel + 0.5) : (int)rel;
    const double d = rel - middle;
    if (degree == 0) {
        weight[0] = 1.0;
    } else if (degree == 1) {
        weight[0] = 1.0 - d;
        weight[1] = d;
    } else if (degree == 2) {
        weight[0] = 0.5 * (0.5 - d) * (0.5 - d);
        weight[1] = 0.75 - d * d;
        weight[2] = 0.5 * (0.5 + d) * (0.5 + d);
    } else {
        const double complement = 1.0 - d;
        const double square = d * d;
        const double cube = square * d;
        weight[0] = complement * complement * complement / 6.0;
        weight[1] = (3.0 * cube - 6.0 * square + 4.0) / 6.0;
        weight[2] = (-3.0 * cube + 3.0 * square + 3.0 * d + 1.0) / 6.0;
        weight[3] = cube / 6.0;
    }
    const int first = middle - degree / 2;
    for (int a = 0; a <= degree; ++a) {
        int k = first + a;
        if (periodic) {
            k = (k + num) % num;
        } else {
            if (k < 0) k = 0;
            if (k > num - 1) k = num - 1;
        }
        node[a] = k;
    }
}

#endif  // MPIFIELD_KERNELS_SPLINE_CELL_CUH
