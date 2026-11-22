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

// The clamped B-spline stencil of `rel` (in cells) on an axis of `num` nodes: knots
// on the nodes, the end ones repeated `degree + 1` times. With `staggered` 0 the
// `degree + 1` B-splines of degree `degree` of the cell; with 1 the `degree`
// rescaled ones D_i = p / (t_(i+p) - t_i) B_(i,p-1), indexed i - 1. Cox-de Boor.
__device__ __forceinline__ void mpifield_clamped_cell(
    double rel, int num, int degree, int staggered, int* node, double* weight
) {
    double left[4], right[4];
    double r = rel;
    if (r < 0.0) r = 0.0;
    if (r > num - 1) r = num - 1;
    int cell = (int)r;
    if (cell > num - 2) cell = num - 2;
    const int span = cell + degree;
    const int order = degree - staggered;
    weight[0] = 1.0;
    for (int k = 1; k <= order; ++k) {
        int low = span + 1 - k - degree;
        int high = span + k - degree;
        if (low < 0) low = 0;
        if (high > num - 1) high = num - 1;
        left[k] = r - low;
        right[k] = high - r;
        double saved = 0.0;
        for (int m = 0; m < k; ++m) {
            const double temp = weight[m] / (right[m + 1] + left[k - m]);
            weight[m] = saved + right[m + 1] * temp;
            saved = left[k - m] * temp;
        }
        weight[k] = saved;
    }
    for (int m = 0; m <= order; ++m) {
        node[m] = cell + m;
        if (staggered) {
            const int i = cell + 1 + m;
            int low = i - degree;
            int high = i;
            if (low < 0) low = 0;
            if (high > num - 1) high = num - 1;
            weight[m] = weight[m] * degree / (double)(high - low);
        }
    }
}

__device__ __forceinline__ void mpifield_spline_cell(
    double rel, int num, int periodic, int degree, int* node, double* weight
) {
    if (periodic == 2) {  // a clamped axis
        mpifield_clamped_cell(rel, num, degree, 0, node, weight);
        return;
    }
    if (periodic == 1) {
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
        if (periodic == 1) {
            k = (k + num) % num;
        } else {
            if (k < 0) k = 0;
            if (k > num - 1) k = num - 1;
        }
        node[a] = k;
    }
}

#endif  // MPIFIELD_KERNELS_SPLINE_CELL_CUH
