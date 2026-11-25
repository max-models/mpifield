"""The C version of scatter_linear_1d against the Pyccel one, on one and several threads."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mpifield.kernels.scatter_linear_1d import kernel

HOST = kernel.host_kernel.kernel
needs_c = pytest.mark.skipif(not HOST.available("c"), reason="no C compiler")


def _case(n, periodic, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-2.5, 3.5, n)  # outside the grid, beyond a period
    positions[:5] = 1.0  # on the upper face
    values = rng.normal(size=n)
    values[::9] = 0.0
    positions[9] = np.nan  # a zero value deposits nothing, whatever its position
    num = 33
    return positions, values, num, (0.0, 1.0 / (num - 1 + periodic), num, periodic)


@needs_c
@pytest.mark.parametrize("periodic", [0, 1])
def test_the_c_version_gives_the_pyccel_field(periodic):
    positions, values, num, args = _case(5000, periodic)
    expected, got = np.full(num, 0.5), np.full(num, 0.5)
    HOST.get("pyccel")(positions, values, expected, *args)
    HOST.get("c")(positions, values, got, *args)
    # equal to round-off: the Pyccel build may contract multiply-adds, the C one does not
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)


@needs_c
def test_a_strided_field_runs_the_pyccel_version():
    positions, values, num, args = _case(5000, 1)
    storage = np.zeros((num, 2))
    kernel(positions, values, storage[:, 0], *args)
    expected = np.zeros(num)
    HOST.get("pyccel")(positions, values, expected, *args)
    np.testing.assert_array_equal(storage[:, 0], expected)
    assert not storage[:, 1].any()


@needs_c
def test_the_c_version_on_several_threads():
    code = textwrap.dedent(
        """
        import numpy as np
        from mpifield.tests.kernels.test_scatter_linear_1d_c import HOST, _case
        for periodic in (0, 1):
            positions, values, num, args = _case(60000, periodic)
            expected, got = np.zeros(num), np.zeros(num)
            HOST.get("pyccel")(positions, values, expected, *args)
            HOST.get("c")(positions, values, got, *args)
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-10)
        """
    )
    env = {**os.environ, "OMP_NUM_THREADS": "3"}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
