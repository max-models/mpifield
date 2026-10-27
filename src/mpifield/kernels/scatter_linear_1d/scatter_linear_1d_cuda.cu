// The CUDA version (GPU) of the scatter_linear_1d kernel; scatter_linear_1d_pyccel.py next to this
// file is the Pyccel version (CPU), with the same arguments.  Loaded by
// mpifield.kernels.catalog: threads visit markers in a grid-stride loop.
//
// Linear deposit onto a 1D grid; on a periodic axis the right neighbour of the
// last node is the first node, on a bounded one a marker outside the grid uses
// the edge cell.  Each block accumulates into a copy of the grid in shared memory
// (launch with shared_mem = 8 * num_gridpoints) and adds it to field once.

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "mpifield/kernels/linear_cell.cuh"

extern "C" __global__
void scatter_linear_1d(
    Array1D<double> positions,
    Array1D<double> values,
    Array1D<double> field,
    double lower_x,
    double cell_size,
    int num_gridpoints,
    int periodic_x
) {
    const long long first = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    const long long stride = blockDim.x * (long long)gridDim.x;
    extern __shared__ double block_field[];
    for (int index = threadIdx.x; index < num_gridpoints; index += blockDim.x) {
        block_field[index] = 0.0;
    }
    __syncthreads();
    for (long long marker = first; marker < positions.shape[0]; marker += stride) {
        const double value = values(marker);
        // Dead markers have zero weight. Do not read their positions, which
        // may be outside the domain (or non-finite), or issue zero atomics.
        if (value == 0.0) continue;
        int left, right;
        double fraction;
        mpifield_linear_cell((positions(marker) - lower_x) / cell_size,
                             num_gridpoints, periodic_x, &left, &right, &fraction);
        cunumpy_atomic_add(&block_field[left], (1.0 - fraction) * value);
        cunumpy_atomic_add(&block_field[right], fraction * value);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < num_gridpoints; index += blockDim.x) {
        const double value = block_field[index];
        if (value != 0.0) cunumpy_atomic_add(&field(index), value);
    }
}
