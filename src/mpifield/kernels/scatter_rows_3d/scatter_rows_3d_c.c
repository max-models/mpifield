/* The C version (CPU) of the scatter_rows_3d kernel: see scatter_rows_3d_pyccel.py.
 *
 * With one OpenMP thread the markers are deposited straight into the field; with more,
 * each thread deposits a contiguous share of them into its own zeroed grid and the grids
 * are summed into the field (no atomics; the order of the sum, hence the last bits,
 * depends on the number of threads). Indices are clamped as integers and the operations
 * are those of the Pyccel version (a build with -march=native fuses multiply-adds).
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline double wrapped(double rel, int64_t num, int64_t periodic) {
    const double n = (double)num;
    if (periodic == 0 || (rel >= 0.0 && rel < n)) return rel;
    return rel - n * floor(rel / n);
}

static inline int64_t left_of(double rel, int64_t last) {
    const int64_t i = (int64_t)rel;
    return i < 0 ? 0 : (i > last ? last : i);
}

static inline int64_t right_of(int64_t left, int64_t num) {
    return left + 1 == num ? 0 : left + 1;
}

static inline double fraction(double rel, int64_t left) {
    const double f = rel - (double)left;
    return f < 0.0 ? 0.0 : (f > 1.0 ? 1.0 : f);
}

static void deposit(int64_t first, int64_t last, const double *restrict px,
                    const double *restrict py, const double *restrict pz,
                    const double *restrict values, double *restrict field,
                    double lower_x, double lower_y, double lower_z,
                    double ix, double iy, double iz, int64_t num_x, int64_t num_y,
                    int64_t num_z, int64_t periodic_x, int64_t periodic_y,
                    int64_t periodic_z) {
    const int64_t last_x = num_x - 2 + periodic_x, last_y = num_y - 2 + periodic_y;
    const int64_t last_z = num_z - 2 + periodic_z;
    for (int64_t m = first; m < last; ++m) {
        const double v = values[m];
        if (v == 0.0) continue;
        const double rx = wrapped((px[m] - lower_x) * ix, num_x, periodic_x);
        const int64_t il = left_of(rx, last_x);
        const double fx = fraction(rx, il);
        const int64_t ir = right_of(il, num_x);
        const double ry = wrapped((py[m] - lower_y) * iy, num_y, periodic_y);
        const int64_t jl = left_of(ry, last_y);
        const double fy = fraction(ry, jl);
        const int64_t jr = right_of(jl, num_y);
        const double rz = wrapped((pz[m] - lower_z) * iz, num_z, periodic_z);
        const int64_t kl = left_of(rz, last_z);
        const double fz = fraction(rz, kl);
        const int64_t kr = right_of(kl, num_z);
        const double gx = 1.0 - fx, gy = 1.0 - fy, gz = 1.0 - fz;
        const double w00 = gx * gy, w01 = gx * fy, w10 = fx * gy, w11 = fx * fy;
        const int64_t r00 = (il * num_y + jl) * num_z, r01 = (il * num_y + jr) * num_z;
        const int64_t r10 = (ir * num_y + jl) * num_z, r11 = (ir * num_y + jr) * num_z;
        field[r00 + kl] += w00 * gz * v;
        field[r00 + kr] += w00 * fz * v;
        field[r01 + kl] += w01 * gz * v;
        field[r01 + kr] += w01 * fz * v;
        field[r10 + kl] += w10 * gz * v;
        field[r10 + kr] += w10 * fz * v;
        field[r11 + kl] += w11 * gz * v;
        field[r11 + kr] += w11 * fz * v;
    }
}

void scatter_rows_3d(
    double *px, const int64_t *px_shape, double *py, const int64_t *py_shape,
    double *pz, const int64_t *pz_shape, double *values, const int64_t *values_shape,
    double *field, const int64_t *field_shape,
    double lower_x, double lower_y, double lower_z,
    double cell_size_x, double cell_size_y, double cell_size_z,
    int64_t num_x, int64_t num_y, int64_t num_z,
    int64_t periodic_x, int64_t periodic_y, int64_t periodic_z) {
    (void)py_shape; (void)pz_shape; (void)values_shape;
    const int64_t n = px_shape[0], nodes = field_shape[0];
    const double ix = 1.0 / cell_size_x, iy = 1.0 / cell_size_y, iz = 1.0 / cell_size_z;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    /* a grid per thread pays only with many markers per node */
    if (threads == 1 || n < 20000 || n < nodes) {
        deposit(0, n, px, py, pz, values, field, lower_x, lower_y, lower_z, ix, iy, iz,
                num_x, num_y, num_z, periodic_x, periodic_y, periodic_z);
        return;
    }
    double *grids = calloc((size_t)threads * (size_t)nodes, sizeof(double));
    if (grids == NULL) {
        deposit(0, n, px, py, pz, values, field, lower_x, lower_y, lower_z, ix, iy, iz,
                num_x, num_y, num_z, periodic_x, periodic_y, periodic_z);
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
        deposit(first, last, px, py, pz, values, grids + (size_t)t * nodes, lower_x,
                lower_y, lower_z, ix, iy, iz, num_x, num_y, num_z, periodic_x,
                periodic_y, periodic_z);
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (int64_t node = 0; node < nodes; ++node) {
            double total = field[node];
            for (int k = 0; k < threads; ++k) total += grids[(size_t)k * nodes + node];
            field[node] = total;
        }
    }
    free(grids);
}
