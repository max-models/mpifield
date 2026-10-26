"""The scatters' parallel path (many markers, per-thread grid copies) against NumPy."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from mpifield.kernels.catalog import KERNELS

NUM_MARKERS = 60000  # above the kernels' MIN_PARALLEL_MARKERS


def _both(name, args, shape):
    fields = {}
    for implementation in ("numpy", "pyccel"):
        field = np.full(shape, 0.5)  # deposits add to what is there
        with xp.kernels.use_host_kernel_implementation(implementation):
            KERNELS[name](*args[:2], field, *args[2:])
        fields[implementation] = field
    return fields["pyccel"], fields["numpy"]


def test_scatter_linear_1d_parallel() -> None:
    rng = np.random.default_rng(0)
    num = 64
    positions = rng.uniform(-0.05, 1.05, NUM_MARKERS)
    values = rng.normal(size=NUM_MARKERS)
    compiled, reference = _both(
        "scatter_linear_1d", (positions, values, 0.0, 1 / (num - 1), num), (num,)
    )
    np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-10)


@pytest.mark.parametrize("periodic", [(0, 0), (1, 0), (1, 1)])
@pytest.mark.parametrize("components", [1, 3])
def test_scatter_linear_2d_parallel(periodic, components) -> None:
    rng = np.random.default_rng(1)
    num_x, num_y = 40, 30
    positions = rng.uniform(-0.05, 1.05, (2, NUM_MARKERS))  # (ndim, N)
    values = rng.normal(size=(components, NUM_MARKERS))  # (components, N)
    args = (positions, values, 0.0, 0.0, 1 / num_x, 1 / num_y, num_x, num_y, *periodic)
    compiled, reference = _both("scatter_linear_2d", args, (num_x, num_y, components))
    np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-10)


@pytest.mark.parametrize("periodic", [(0, 0, 0), (0, 1, 1)])
@pytest.mark.parametrize("components", [1, 3])
def test_scatter_linear_3d_parallel(periodic, components) -> None:
    rng = np.random.default_rng(2)
    nx, ny, nz = 20, 12, 12
    positions = rng.uniform(-0.05, 1.05, (3, NUM_MARKERS))  # (ndim, N)
    values = rng.normal(size=(components, NUM_MARKERS))  # (components, N)
    args = (
        positions,
        values,
        0.0,
        0.0,
        0.0,
        1 / nx,
        1 / ny,
        1 / nz,
        nx,
        ny,
        nz,
        *periodic,
    )
    compiled, reference = _both("scatter_linear_3d", args, (nx * ny * nz, components))
    np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-10)


def _sorted_by_x(positions, values):
    order = np.argsort(positions[0], kind="stable")
    return positions[:, order].copy(), values[:, order].copy()


def _arguments_2d(positions, values, periodic, num_x=40, num_y=30):
    return (positions, values, 0.0, 0.0, 1 / num_x, 1 / num_y, num_x, num_y, *periodic)


def _arguments_3d(positions, values, periodic, shape=(20, 12, 12)):
    nx, ny, nz = shape
    return (positions, values, 0.0, 0.0, 0.0, 1 / nx, 1 / ny, 1 / nz, *shape, *periodic)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("periodic_x", [0, 1])
@pytest.mark.parametrize("ordering", ["sorted", "unsorted"])
def test_banded_buffers_match_numpy(ndim, periodic_x, ordering) -> None:
    """Sorted markers give each thread a narrow band of x-planes; unsorted the whole grid."""
    rng = np.random.default_rng(3)
    positions = rng.uniform(-0.05, 1.05, (ndim, NUM_MARKERS))
    values = rng.normal(size=(2, NUM_MARKERS))
    if ordering == "sorted":
        positions, values = _sorted_by_x(positions, values)
    periodic = (periodic_x, 1) if ndim == 2 else (periodic_x, 1, 0)
    if ndim == 2:
        args, shape = _arguments_2d(positions, values, periodic), (40, 30, 2)
    else:
        args, shape = _arguments_3d(positions, values, periodic), (20 * 12 * 12, 2)
    compiled, reference = _both(f"scatter_linear_{ndim}d", args, shape)
    np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-10)


@pytest.mark.parametrize("ndim", [2, 3])
def test_periodic_band_past_the_last_plane_folds_back_to_plane_zero(ndim) -> None:
    """Sorted markers in the last periodic cell deposit on plane num_x - 1 and plane 0."""
    rng = np.random.default_rng(4)
    positions = rng.uniform(0.0, 1.0, (ndim, NUM_MARKERS))
    # most markers in the last x-cell, so the last thread's band ends at plane num_x
    positions[0, : NUM_MARKERS // 2] = rng.uniform(0.96, 1.0, NUM_MARKERS // 2)
    values = rng.normal(size=(1, NUM_MARKERS))
    positions, values = _sorted_by_x(positions, values)
    if ndim == 2:
        args, shape = _arguments_2d(positions, values, (1, 1)), (40, 30, 1)
    else:
        args, shape = _arguments_3d(positions, values, (1, 1, 1)), (20 * 12 * 12, 1)
    compiled, reference = _both(f"scatter_linear_{ndim}d", args, shape)
    np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-10)


@pytest.mark.parametrize("ndim", [2, 3])
def test_chunks_with_only_zero_values_and_reproducible_sums(ndim) -> None:
    """Chunks of dead (zero-value) markers get no band; repeated calls agree bitwise."""
    rng = np.random.default_rng(5)
    positions = rng.uniform(0.0, 1.0, (ndim, NUM_MARKERS))
    values = rng.normal(size=(1, NUM_MARKERS))
    values[:, : NUM_MARKERS // 2] = 0.0  # the first threads' chunks deposit nothing
    positions, values = _sorted_by_x(positions, values)
    values[:, : NUM_MARKERS // 2] = 0.0
    if ndim == 2:
        args, shape = _arguments_2d(positions, values, (0, 0)), (40, 30, 1)
    else:
        args, shape = _arguments_3d(positions, values, (0, 0, 0)), (20 * 12 * 12, 1)
    compiled, reference = _both(f"scatter_linear_{ndim}d", args, shape)
    np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-10)
    again, _ = _both(f"scatter_linear_{ndim}d", args, shape)
    np.testing.assert_array_equal(again, compiled)
