"""The Metal gather and scatter kernels on the Apple-silicon GPU, against the CPU.

The GPU computes in float32 and its float atomics add in another order than the CPU,
so the results agree to about 1e-5 of the largest value. Skipped without MLX and a
Metal GPU.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from cunumpy.kernels import metal_available

from mpifield.kernels.catalog import KERNELS

pytestmark = pytest.mark.skipif(
    not metal_available(), reason="MLX with a Metal GPU is not available"
)

N = 4000
LOWER = (0.1, -0.2, 0.3)
CELL = (0.05, 0.04, 0.06)
NUM = (12, 9, 7)


def _metal(name):
    module = importlib.import_module(f"mpifield.kernels.{name}.{name}_metal")
    return getattr(module, name)


def _close(gpu, cpu):
    scale = max(float(np.abs(cpu).max()), 1.0)
    np.testing.assert_allclose(gpu / scale, cpu / scale, rtol=0.0, atol=3e-5)


def _positions(ndim, seed=0):
    rng = np.random.default_rng(seed)
    # from a bit outside the grid on both sides: bounded axes clamp, periodic wrap
    span = [(NUM[a] - 1) * CELL[a] for a in range(ndim)]
    # component-major (ndim, N)
    return np.stack(
        [LOWER[a] + rng.uniform(-0.2, 1.2, N) * span[a] for a in range(ndim)], axis=0
    )


def _axis_args(ndim, periodic):
    return (
        *LOWER[:ndim],
        *CELL[:ndim],
        *NUM[:ndim],
        *[int(p) for p in periodic],
    )


PERIODIC = {
    1: [(0,), (1,)],
    2: [(0, 0), (1, 0), (0, 1), (1, 1)],
    3: [(0, 0, 0), (1, 1, 1), (1, 0, 1)],
}


def _grid_shape(ndim, ncomp):
    if ndim == 3:
        return (NUM[0] * NUM[1] * NUM[2], ncomp)
    return (*NUM[:ndim], ncomp)


@pytest.mark.parametrize(
    ("ndim", "periodic"), [(d, p) for d, ps in PERIODIC.items() for p in ps]
)
def test_gather_linear(ndim, periodic):
    name = f"gather_linear_{ndim}d"
    positions = _positions(ndim)
    field = np.random.default_rng(1).normal(size=_grid_shape(ndim, 3))
    results = []
    for function in (KERNELS[name], _metal(name)):
        result = np.zeros((3, N))  # (components, N)
        function(positions, field, result, *_axis_args(ndim, periodic))
        results.append(result)
    _close(results[1], results[0])


@pytest.mark.parametrize(
    ("ndim", "periodic"), [(d, p) for d, ps in PERIODIC.items() for p in ps if d > 1]
)
def test_scatter_linear(ndim, periodic):
    name = f"scatter_linear_{ndim}d"
    positions = _positions(ndim)
    values = np.random.default_rng(2).normal(size=(2, N))  # (components, N)
    results = []
    for function in (KERNELS[name], _metal(name)):
        field = np.full(_grid_shape(ndim, 2), 0.25)  # accumulates into existing values
        function(positions, values, field, *_axis_args(ndim, periodic))
        results.append(field)
    _close(results[1], results[0])


def test_scatter_linear_1d_skips_dead_markers():
    name = "scatter_linear_1d"
    rng = np.random.default_rng(3)
    positions = LOWER[0] + rng.uniform(-0.2, 1.2, N) * (NUM[0] - 1) * CELL[0]
    values = rng.normal(size=N)
    values[::3] = 0.0
    positions[::3] = np.nan  # dead markers may sit anywhere
    results = []
    for function in (KERNELS[name], _metal(name)):
        field = np.full(NUM[0], 0.25)
        function(positions, values, field, LOWER[0], CELL[0], NUM[0], 0)
        results.append(field)
    assert np.isfinite(results[1]).all()
    _close(results[1], results[0])


@pytest.mark.parametrize(
    ("ndim", "periodic"), [(d, p) for d, ps in PERIODIC.items() for p in ps]
)
def test_gather_quadratic(ndim, periodic):
    name = f"gather_quadratic_{ndim}d"
    positions = _positions(ndim, seed=4)
    field = np.random.default_rng(5).normal(size=(*NUM[:ndim], 3))
    results = []
    for function in (KERNELS[name], _metal(name)):
        result = np.zeros((3, N))  # (components, N)
        function(positions, field, result, *_axis_args(ndim, periodic))
        results.append(result)
    _close(results[1], results[0])


@pytest.mark.parametrize(
    ("ndim", "periodic"), [(d, p) for d, ps in PERIODIC.items() for p in ps]
)
def test_scatter_quadratic(ndim, periodic):
    name = f"scatter_quadratic_{ndim}d"
    positions = _positions(ndim, seed=6)
    values = np.random.default_rng(7).normal(size=(2, N))  # (components, N)
    values[:, ::3] = 0.0
    positions[:, ::3] = np.nan  # dead markers may sit anywhere
    results = []
    for function in (KERNELS[name], _metal(name)):
        field = np.full((*NUM[:ndim], 2), 0.25)  # accumulates into existing values
        function(positions, values, field, *_axis_args(ndim, periodic))
        results.append(field)
    assert np.isfinite(results[1]).all()
    _close(results[1], results[0])
