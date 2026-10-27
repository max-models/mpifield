// mpifield/kernels/quadratic_cell.cuh: the stencil of a position along one grid
// axis for quadratic B-spline weights.
//
// Shared by the CUDA versions of the quadratic deposit and gather kernels, and
// the same arithmetic as their Pyccel versions: the nearest node c and its two
// neighbours, with the offset d = rel - c in [-1/2, 1/2] and the weights
// (1/2 - d)^2 / 2, 3/4 - d^2, (1/2 + d)^2 / 2.  On a periodic axis the `num`
// nodes cover `num` cells and a position wraps into the period; on a bounded
// axis it is clamped to [0, num - 1] and a stencil node past the edge is the
// edge node.

#ifndef MPIFIELD_KERNELS_QUADRATIC_CELL_CUH
#define MPIFIELD_KERNELS_QUADRATIC_CELL_CUH

__device__ __forceinline__ void mpifield_quadratic_cell(
    double rel, int num, int periodic, int* node, double* weight
) {
    int c;
    if (periodic) {
        rel = rel - num * floor(rel / num);
        c = (int)(rel + 0.5);
        node[0] = (c + num - 1) % num;
        node[1] = c % num;
        node[2] = (c + 1) % num;
    } else {
        if (rel < 0.0) rel = 0.0;
        if (rel > num - 1) rel = num - 1;
        c = (int)(rel + 0.5);
        node[0] = c > 0 ? c - 1 : 0;
        node[1] = c;
        node[2] = c < num - 1 ? c + 1 : num - 1;
    }
    const double d = rel - c;
    weight[0] = 0.5 * (0.5 - d) * (0.5 - d);
    weight[1] = 0.75 - d * d;
    weight[2] = 0.5 * (0.5 + d) * (0.5 + d);
}

#endif  // MPIFIELD_KERNELS_QUADRATIC_CELL_CUH
