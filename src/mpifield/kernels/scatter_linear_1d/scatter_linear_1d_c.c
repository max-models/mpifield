/* The C version (CPU) of the scatter_linear_1d kernel: see scatter_linear_1d_pyccel.py.
 *
 * The same nodes, weights and operations, without fused multiply-adds (the same field as
 * a Pyccel build that does not contract them, on one thread). On several OpenMP threads
 * each deposits a contiguous share of the markers into its own grid and the grids are
 * added to the field, as the Pyccel version does.
 * Taken for C-contiguous arrays; cunumpy runs the Pyccel version for strided ones.
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

/* MIN_PARALLEL_MARKERS of the Pyccel version */
#define MIN_PARALLEL 20000

static void deposit(int64_t first, int64_t last, const double *restrict positions,
                    const double *restrict values, double *restrict grid, double lower_x,
                    double cell_size, int64_t num, int64_t periodic) {
    const double n = (double)num;
    for (int64_t m = first; m < last; ++m) {
        const double value = values[m];
        if (value == 0.0) continue; /* a dead marker, whatever its position */
        double rel = (positions[m] - lower_x) / cell_size;
        int64_t left, right;
        if (periodic == 1) {
            /* mpifield's periodic_position, left_node and right_node */
            if (!(rel >= 0.0 && rel < n)) rel = rel - n * floor(rel / n);
            left = (int64_t)rel >= num ? num - 1 : (int64_t)rel;
            right = left + 1 == num ? 0 : left + 1;
        } else {
            const int64_t i = rel >= 0.0 ? (int64_t)rel : 0;
            left = i > num - 2 ? num - 2 : i;
            right = left + 1;
        }
        const double d = rel - (double)left;
        const double fraction = d < 0.0 ? 0.0 : (d > 1.0 ? 1.0 : d);
        grid[left] += (1.0 - fraction) * value;
        grid[right] += fraction * value;
    }
}

void scatter_linear_1d(double *positions, const int64_t *positions_shape, double *values,
                       const int64_t *values_shape, double *field,
                       const int64_t *field_shape, double lower_x, double cell_size,
                       int64_t num_gridpoints, int64_t periodic_x) {
    (void)values_shape;
    const int64_t n = positions_shape[0], nodes = field_shape[0];
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    if (threads == 1 || n < MIN_PARALLEL || n < nodes) {
        deposit(0, n, positions, values, field, lower_x, cell_size, num_gridpoints,
                periodic_x);
        return;
    }
    double *grids = calloc((size_t)threads * (size_t)num_gridpoints, sizeof(double));
    if (grids == NULL) {
        deposit(0, n, positions, values, field, lower_x, cell_size, num_gridpoints,
                periodic_x);
        return;
    }
    const int64_t share = (n + threads - 1) / threads;
    #pragma omp parallel num_threads(threads)
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        const int64_t first = t * share < n ? t * share : n;
        const int64_t last = first + share < n ? first + share : n;
        deposit(first, last, positions, values, grids + (size_t)t * num_gridpoints,
                lower_x, cell_size, num_gridpoints, periodic_x);
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (int64_t node = 0; node < nodes; ++node) {
            double total = 0.0;
            for (int k = 0; k < threads; ++k) total += grids[(size_t)k * num_gridpoints + node];
            field[node] += total;
        }
    }
    free(grids);
}
