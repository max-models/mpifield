"""The kernel catalog: one folder per kernel, and the compiled Pyccel versions.

Each compiled Pyccel version is compared with the NumPy version
(``<name>_numpy.py``) that runs without Pyccel (forced with
``cunumpy.kernels.use_host_kernel_implementation("numpy")``). The CUDA versions are
compared with the same CPU code in ``test_cuda_emulated.py``.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import cunumpy as xp
import numpy as np
import pytest

from mpifield import LINEAR, QUADRATIC, Field
from mpifield.kernels.catalog import KERNEL_NAMES, KERNELS, metal_kernel


def _folder(name):
    return Path(importlib.import_module(f"mpifield.kernels.{name}").__file__).parent


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_each_kernel_has_a_folder_with_all_versions(name):
    folder = _folder(name)
    pyccel = folder / f"{name}_pyccel.py"
    cuda = folder / f"{name}_cuda.cu"
    assert pyccel.is_file() and cuda.is_file()
    functions = {
        node.name: node
        for node in ast.parse(pyccel.read_text()).body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }
    # one kernel per file, named like its folder (``_helpers`` are private)
    assert list(functions) == [name]
    assert re.search(rf"__global__\s+void\s+{name}\s*\(", cuda.read_text())
    assert (folder / f"{name}_numpy.py").is_file()
    assert (folder / f"{name}_metal.py").is_file()


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_each_folder_declares_one_kernel_for_both_backends(name):
    kernel = importlib.import_module(f"mpifield.kernels.{name}").kernel
    assert isinstance(kernel, xp.kernels.Kernel)
    assert KERNELS[name] is kernel
    assert kernel.name == name
    assert kernel.has_cuda
    assert kernel.dispatch == "arrays"  # NumPy arrays -> Pyccel, CuPy -> CUDA
    # Pyccel, NumPy and CUDA take the same parameters, in the same order
    kernel.check_signature()
    assert kernel.host_parameters() == [
        param.name for param in kernel.cuda_kernel.signature
    ]


def test_metal_kernel_rejects_unknown_names():
    with pytest.raises(KeyError, match="no Metal version"):
        metal_kernel("push_boris_1d")


def _compiled(name):
    if not KERNELS[name].host_kernel.kernel.available("pyccel"):
        pytest.skip("Pyccel kernels unavailable")


def _field(ndim, periodic=None, components=1):
    shape = (13, 9, 7)[:ndim]
    return Field(
        ndim=ndim,
        num_components=components,
        lower_bound=(-0.2, 0.1, 0.0)[:ndim],
        upper_bound=(1.3, 0.7, 0.9)[:ndim],
        num_gridpoints=shape,
        periodic=periodic,
    )


def _points(rng, field, n=2000):
    lower = np.asarray(field.lower_bound)
    upper = np.asarray(field.upper_bound)
    # include points slightly outside: clamped on bounded axes, wrapped on periodic;
    # component-major (ndim, N)
    return (lower - 0.05)[:, None] + (upper - lower + 0.1)[:, None] * rng.random(
        (len(lower), n)
    )


@pytest.mark.parametrize(
    ("ndim", "periodic", "components"),
    [
        (1, None, 1),
        (2, (False, False), 1),
        (2, (False, True), 2),
        (2, (True, True), 3),
        (3, (False, False, False), 1),
        (3, (True, False, True), 2),
        (3, (True, True, True), 3),
    ],
)
@pytest.mark.parametrize("weighting", [LINEAR, QUADRATIC], ids=lambda w: w.name)
def test_compiled_deposits_match_numpy(ndim, periodic, components, weighting):
    _compiled(f"scatter_{weighting.name}_{ndim}d")
    rng = np.random.default_rng(1)
    compiled = _field(ndim, periodic, components)
    points = _points(rng, compiled)
    values = rng.uniform(0.5, 1.5, (components, points.shape[1]))
    compiled.scatter(points, values, weighting)

    fallback = _field(ndim, periodic, components)
    with xp.kernels.use_host_kernel_implementation("numpy"):  # as without Pyccel
        fallback.scatter(points, values, weighting)
    np.testing.assert_allclose(
        xp.to_numpy(compiled.local_with_halos),
        xp.to_numpy(fallback.local_with_halos),
        rtol=1e-13,
        atol=1e-13,
    )
    assert np.isclose(float(compiled.local_with_halos.sum()), values.sum())


@pytest.mark.parametrize(
    ("ndim", "periodic"),
    [
        (1, None),
        (1, (True,)),
        (2, (False, True)),
        (2, (True, True)),
        (3, (False, False, False)),
        (3, (True, False, True)),
    ],
)
@pytest.mark.parametrize("weighting", [LINEAR, QUADRATIC], ids=lambda w: w.name)
def test_compiled_gathers_match_numpy(ndim, periodic, weighting):
    _compiled(f"gather_{weighting.name}_{ndim}d")
    rng = np.random.default_rng(2)
    field = _field(ndim, periodic, components=2)
    field.local_with_halos[...] = rng.normal(size=field.local_with_halos.shape)
    points = _points(rng, field)
    compiled = np.asarray(xp.to_numpy(field.gather(points, weighting)))
    with xp.kernels.use_host_kernel_implementation("numpy"):
        fallback = np.asarray(xp.to_numpy(field.gather(points, weighting)))
    np.testing.assert_allclose(compiled, fallback, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("weighting", [LINEAR, QUADRATIC], ids=lambda w: w.name)
def test_deposit_and_gather_are_transposes(ndim, periodic, weighting):
    """sum_i g(x_i) q_i = sum_n G_n rho_n: each weighting's two kernels are adjoint.

    Also at the edges of bounded axes, where the quadratic stencil folds back.
    """
    rng = np.random.default_rng(7)
    field = _field(ndim, (periodic,) * ndim)
    density = _field(ndim, (periodic,) * ndim)
    field.local_with_halos[...] = rng.normal(size=field.local_with_halos.shape)
    lower = np.asarray(field.lower_bound)
    upper = np.asarray(field.upper_bound)
    points = lower[:, None] + (upper - lower)[:, None] * rng.random((ndim, 500))
    charges = rng.normal(size=500)

    density.scatter(points, charges, weighting)
    gathered = np.asarray(xp.to_numpy(field.gather(points, weighting)))[0]

    assert np.dot(gathered, charges) == pytest.approx(
        float(np.sum(field.local_with_halos * density.local_with_halos)), rel=1e-12
    )
