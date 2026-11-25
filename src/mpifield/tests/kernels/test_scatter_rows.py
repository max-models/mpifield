"""scatter_rows_3d (C, Pyccel, NumPy) against scatter_linear_3d, and LINEAR's choice of it."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mpifield import LINEAR, BlockGrid
from mpifield.kernels.catalog import KERNELS
from mpifield.kernels.scatter_rows_3d import kernel as scatter_rows_3d

SHAPE = (7, 6, 5)


def _case(periodic, n=30000, seed=0):
    rng = np.random.default_rng(seed)
    lower, spacing = np.array([0.0, -1.0, 2.0]), np.array([0.25, 0.5, 0.1])
    cells = np.array(SHAPE) - 1 + np.array(periodic)
    upper = lower + spacing * cells
    # inside, outside, more than a period away, and on the upper faces
    positions = rng.uniform(lower - 2.0 * spacing * cells, upper + 0.3, (n, 3)).T.copy()
    positions[:, : n // 2] = rng.uniform(lower, upper, (n // 2, 3)).T
    positions[:, :10] = upper[:, None]
    values = rng.normal(size=n)
    values[::7] = 0.0
    positions[:, 7] = np.nan  # a zero value deposits nothing, whatever its position
    geometry = (*lower, *spacing, *SHAPE, *(int(p) for p in periodic))
    return positions, values, geometry


def _reference(positions, values, geometry):
    field = np.full((int(np.prod(SHAPE)), 1), 0.5)
    KERNELS["scatter_linear_3d"](positions, values.reshape(1, -1), field, *geometry)
    return field[:, 0]


@pytest.mark.parametrize("implementation", ["c", "pyccel", "numpy"])
@pytest.mark.parametrize("periodic", [(0, 0, 0), (0, 1, 1), (1, 1, 1)])
def test_every_version_matches_scatter_linear_3d(implementation, periodic):
    host = scatter_rows_3d.host_kernel.kernel
    if not host.available(implementation):
        pytest.skip(f"no {implementation} version: {host.errors.get(implementation)}")
    positions, values, geometry = _case(periodic)
    expected = _reference(positions, values, geometry)
    field = np.full(int(np.prod(SHAPE)), 0.5)
    host.get(implementation)(*positions, values, field, *geometry)
    np.testing.assert_allclose(field, expected, rtol=1e-12, atol=1e-12)


def test_linear_takes_the_rows_kernel_and_agrees():
    grid = BlockGrid.from_geometry((0.0, -1.0, 2.0), (0.25, 0.5, 0.1), SHAPE, (0, 1, 1))
    positions, values, _ = _case((0, 1, 1))
    storage = np.zeros((3, positions.shape[1] + 50))
    storage[:, : positions.shape[1]] = positions  # strided rows, as a marker store's
    field = np.zeros((*SHAPE, 1))
    LINEAR.scatter_block(storage[:, : positions.shape[1]], values, field, grid)
    expected = _reference(positions, values, grid.geometry)
    np.testing.assert_allclose(
        field.reshape(-1), expected - 0.5, rtol=1e-12, atol=1e-12
    )


@pytest.mark.skipif(
    not scatter_rows_3d.host_kernel.kernel.available("c"), reason="no C compiler"
)
def test_the_c_version_on_several_threads():
    code = textwrap.dedent(
        """
        import numpy as np
        from mpifield.tests.kernels.test_scatter_rows import _case, _reference
        from mpifield.kernels.scatter_rows_3d import kernel
        positions, values, geometry = _case((0, 1, 1), n=60000)
        expected = _reference(positions, values, geometry)
        field = np.full(expected.shape, 0.5)
        kernel.host_kernel.kernel.get("c")(*positions, values, field, *geometry)
        np.testing.assert_allclose(field, expected, rtol=1e-12, atol=1e-12)
        """
    )
    env = {**os.environ, "OMP_NUM_THREADS": "3"}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
