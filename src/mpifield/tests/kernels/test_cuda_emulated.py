"""The CUDA kernels' arithmetic, checked on the CPU without a GPU.

Each kernel body is compiled as C++ with the CUDA builtins stubbed and called
once per thread index, serially (atomics are then plain additions).  The
results are compared with mpifield's CPU kernels, including periodic axes and
markers outside the grid.  This checks the indexing and weights, not the
concurrency; the GPU tests in ``test_cuda_kernels.py`` cover a real launch.
Skipped without a C++ compiler.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cunumpy as xp
import numpy as np
import pytest

import mpifield
from mpifield import LINEAR, QUADRATIC, BlockGrid
from mpifield.kernels.catalog import KERNELS

COMPILER = shutil.which("c++")
# the directory containing the mpifield package, for #include "mpifield/..."
SOURCE_ROOT = Path(mpifield.__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(COMPILER is None, reason="no C++ compiler")

STUBS = r"""
#include <cmath>
#include <cstdio>
using std::floor;
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
struct dim3_ { unsigned x, y, z; };
static dim3_ threadIdx, blockDim, blockIdx, gridDim;
#define __shared__
inline void __syncthreads() {} // every emulated block has one thread
double block_field[6144];
inline double atomicAdd(double* p, double v) { double o = *p; *p += v; return o; }
inline float atomicAdd(float* p, float v) { float o = *p; *p += v; return o; }
inline int atomicAdd(int* p, int v) { int o = *p; *p += v; return o; }
inline unsigned atomicAdd(unsigned* p, unsigned v) { unsigned o = *p; *p += v; return o; }
inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) { unsigned long long o = *p; *p += v; return o; }
inline double __longlong_as_double(long long) { return 0; }
inline long long __double_as_longlong(double) { return 0; }
inline unsigned long long atomicCAS(unsigned long long* p, unsigned long long compare, unsigned long long v) { unsigned long long o = *p; if (o == compare) *p = v; return o; }
"""


#: The C type of each array dtype: masks are ``char`` arrays on the GPU.
C_TYPES = {"f": "double", "b": "char", "i": "long long"}


def _c_type(array):
    """The C type of ``array``'s elements (``int`` for int32, e.g. stencil row maps)."""
    if array.dtype == np.int32:
        return "int"
    return C_TYPES[array.dtype.kind]


def _c_array(name, values):
    values = np.asarray(values)
    if values.dtype.kind == "f":
        flat = ", ".join(
            "NAN" if np.isnan(v) else repr(float(v)) for v in np.ravel(values)
        )
    else:
        flat = ", ".join(str(int(v)) for v in np.ravel(values))
    return f"static {_c_type(values)} {name}[] = {{{flat}}};\n"


def _view(name, array, ndim):
    shape = ", ".join(str(n) for n in array.shape)
    strides = ", ".join(str(s // array.itemsize) for s in array.strides)
    c_type = _c_type(array)
    return (
        f"Array{ndim}D<{c_type}> {name}_view{{{name}, {{{shape}}}, {{{strides}}}}};\n"
    )


def _run(tmp_path, kernel, arrays, views, call, outputs):
    """Compile `kernel` with a main that calls it for each thread and prints `outputs`."""
    n_threads, call = call
    body = "".join(_c_array(name, a) for name, a in arrays.items())
    body += f"int main() {{ blockDim.x = 1; gridDim.x = {n_threads};\n"
    body += "".join(_view(name, arrays[name], ndim) for name, ndim in views.items())
    body += (
        f"for (long long t = 0; t < {n_threads}; ++t) "
        f"{{ blockIdx.x = (unsigned)t; {call}; }}\n"
    )
    body += "".join(
        f"for (long long i = 0; i < {arrays[name].size}; ++i) "
        f'printf("%.17g\\n", (double){name}[i]);\n'
        for name in outputs
    )
    body += "return 0; }\n"
    source = tmp_path / "kernel.cpp"
    exe = tmp_path / "kernel"
    source.write_text(STUBS + kernel.source.replace('extern "C"', "") + body)
    subprocess.run(
        [
            COMPILER,
            "-std=c++17",
            "-O1",
            f"-I{xp.cuda.cuda_include_dir()}",
            f"-I{SOURCE_ROOT}",  # mpifield/kernels/linear_cell.cuh
            str(source),
            "-o",
            str(exe),
        ],
        check=True,
    )
    printed = subprocess.run(
        [str(exe)], capture_output=True, text=True, check=True
    ).stdout
    values = np.array([float(v) for v in printed.split()])
    results, start = [], 0
    for name in outputs:
        size = arrays[name].size
        results.append(values[start : start + size].reshape(arrays[name].shape))
        start += size
    return results


@pytest.mark.parametrize("periodic", [0, 1])
@pytest.mark.parametrize("nx", [5, 65, 256])
def test_1d_deposit_grid_stride_and_zero_weights(tmp_path, nx, periodic):
    """Blocks process multiple markers and skip invalid inactive positions."""
    from mpifield.kernels.scatter_linear_1d.scatter_linear_1d_numpy import (
        scatter_linear_1d,
    )

    rng = np.random.default_rng(12)
    positions = rng.uniform(-0.1, 1.1, 517)
    values = rng.normal(size=positions.size)
    values[::3] = 0.0
    positions[::3] = np.nan
    initial = rng.normal(size=nx)
    expected = initial.copy()
    active = values != 0.0
    scatter_linear_1d(
        positions[active], values[active], expected, 0.0, 1.0 / nx, nx, periodic
    )
    (actual,) = _run(
        tmp_path,
        KERNELS["scatter_linear_1d"].cuda_kernel,
        {"positions": positions, "values": values, "field": initial},
        {"positions": 1, "values": 1, "field": 1},
        (
            7,
            "scatter_linear_1d(positions_view, values_view, field_view, "
            f"0.0, {1.0 / nx!r}, {nx}, {periodic})",
        ),
        ["field"],
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.sum(actual - initial) == pytest.approx(np.sum(values))


def _grid_case(seed):
    rng = np.random.default_rng(seed)
    nx, ny, ncomp, n = 9, 7, 2, 300
    lower, spacing = np.array([0.0, -1.0]), np.array([0.25, 0.5])
    upper = lower + spacing * np.array([nx, ny])
    # component-major (2, N); some points outside the grid
    points = rng.uniform(lower - 0.3, upper + 0.3, (n, 2)).T.copy()
    return {
        "nx": nx,
        "ny": ny,
        "ncomp": ncomp,
        "n": n,
        "lower": lower,
        "spacing": spacing,
        "points": points,
        "values": rng.normal(size=(ncomp, n)),
        "field": rng.normal(size=(nx, ny, ncomp)),
    }


def _block(case, periodic):
    return BlockGrid.from_geometry(
        case["lower"], case["spacing"], (case["nx"], case["ny"]), periodic
    )


def _geometry_args(case, periodic):
    lo, sp = (float(v) for v in case["lower"]), (float(v) for v in case["spacing"])
    lo, sp = list(lo), list(sp)
    return (
        f"{lo[0]!r}, {lo[1]!r}, {sp[0]!r}, {sp[1]!r}, {case['nx']}, {case['ny']}, "
        f"{int(periodic[0])}, {int(periodic[1])}"
    )


@pytest.mark.parametrize(
    "periodic", [(False, False, False), (True, False, True), (True, True, True)]
)
def test_scatter_linear_3d_matches_cpu_kernel(tmp_path, periodic):
    rng = np.random.default_rng(7)
    n, shape, ncomp = 200, (5, 4, 6), 2
    lower, spacing = np.array([0.0, -1.0, 2.0]), np.array([0.25, 0.5, 0.1])
    upper = lower + spacing * np.array(shape)
    # component-major (3, N); some points outside the grid
    points = rng.uniform(lower - 0.2, upper + 0.2, (n, 3)).T.copy()
    values = rng.normal(size=(ncomp, n))
    expected = np.zeros((*shape, ncomp))
    with xp.kernels.use_host_kernel_implementation("numpy"):  # the CPU reference
        LINEAR.scatter_block(
            points,
            values,
            expected,
            BlockGrid.from_geometry(lower, spacing, shape, periodic),
        )
    arrays = {
        "positions": points,
        "values": values,
        "field": np.zeros((int(np.prod(shape)), ncomp)),
    }
    geometry = ", ".join(
        [repr(float(v)) for v in lower]
        + [repr(float(v)) for v in spacing]
        + [str(v) for v in shape]
        + [str(int(p)) for p in periodic]
    )
    call = (
        n,
        f"scatter_linear_3d(positions_view, values_view, field_view, {geometry})",
    )
    (result,) = _run(
        tmp_path,
        KERNELS["scatter_linear_3d"].cuda_kernel,
        arrays,
        {"positions": 2, "values": 2, "field": 2},
        call,
        ["field"],
    )
    np.testing.assert_allclose(
        result.reshape(expected.shape), expected, rtol=1e-12, atol=1e-12
    )
    assert np.isclose(result.sum(), values.sum())  # charge is conserved


PERIODIC = [(False, False), (True, False), (False, True), (True, True)]


@pytest.mark.parametrize("periodic", PERIODIC)
def test_scatter_linear_2d_matches_cpu_kernel(tmp_path, periodic):
    case = _grid_case(1)
    expected = np.zeros_like(case["field"])
    LINEAR.scatter_block(
        case["points"], case["values"], expected, _block(case, periodic)
    )
    arrays = {
        "positions": case["points"],
        "values": case["values"],
        "field": np.zeros_like(case["field"]),
    }
    call = (
        case["n"],
        "scatter_linear_2d(positions_view, values_view, field_view, "
        + _geometry_args(case, periodic)
        + ")",
    )
    (result,) = _run(
        tmp_path,
        KERNELS["scatter_linear_2d"].cuda_kernel,
        arrays,
        {"positions": 2, "values": 2, "field": 3},
        call,
        ["field"],
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("periodic", PERIODIC)
def test_gather_linear_2d_matches_cpu_kernel(tmp_path, periodic):
    case = _grid_case(2)
    expected = np.zeros((case["ncomp"], case["n"]))  # (components, N)
    LINEAR.gather_block(case["points"], case["field"], expected, _block(case, periodic))
    arrays = {
        "positions": case["points"],
        "field": case["field"],
        "result": np.zeros_like(expected),
    }
    call = (
        case["n"],
        "gather_linear_2d(positions_view, field_view, result_view, "
        + _geometry_args(case, periodic)
        + ")",
    )
    (result,) = _run(
        tmp_path,
        KERNELS["gather_linear_2d"].cuda_kernel,
        arrays,
        {"positions": 2, "field": 3, "result": 2},
        call,
        ["result"],
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)


def test_gather_linear_1d_matches_cpu_kernel(tmp_path):
    rng = np.random.default_rng(4)
    n, nx, ncomp = 250, 11, 2
    lower, dx = -0.5, 0.2
    points = rng.uniform(lower - 0.4, lower + nx * dx + 0.4, (1, n))  # some outside
    field = rng.normal(size=(nx, ncomp))
    expected = np.zeros((ncomp, n))  # (components, N)
    LINEAR.gather_block(
        points, field, expected, BlockGrid.from_geometry((lower,), (dx,), (nx,))
    )
    arrays = {"positions": points, "field": field, "result": np.zeros_like(expected)}
    call = (
        n,
        (
            f"gather_linear_1d(positions_view, field_view, result_view, {lower!r}, "
            f"{dx!r}, {nx}, 0)"
        ),
    )
    (result,) = _run(
        tmp_path,
        KERNELS["gather_linear_1d"].cuda_kernel,
        arrays,
        {"positions": 2, "field": 2, "result": 2},
        call,
        ["result"],
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "periodic", [(False, False, False), (True, False, True), (True, True, True)]
)
def test_gather_linear_3d_matches_cpu_kernel(tmp_path, periodic):
    rng = np.random.default_rng(6)
    n, shape, ncomp = 200, (5, 4, 6), 3
    lower, spacing = np.array([0.0, -1.0, 2.0]), np.array([0.25, 0.5, 0.1])
    upper = lower + spacing * np.array(shape)
    points = rng.uniform(lower - 0.2, upper + 0.2, (n, 3)).T.copy()  # (3, N)
    field = rng.normal(size=(*shape, ncomp))
    expected = np.zeros((ncomp, n))  # (components, N)
    LINEAR.gather_block(
        points,
        field,
        expected,
        BlockGrid.from_geometry(lower, spacing, shape, periodic),
    )
    flat = np.ascontiguousarray(field.reshape(-1, ncomp))
    arrays = {"positions": points, "field": flat, "result": np.zeros_like(expected)}
    geometry = ", ".join(
        [repr(float(v)) for v in lower]
        + [repr(float(v)) for v in spacing]
        + [str(v) for v in shape]
        + [str(int(p)) for p in periodic]
    )
    call = (n, f"gather_linear_3d(positions_view, field_view, result_view, {geometry})")
    (result,) = _run(
        tmp_path,
        KERNELS["gather_linear_3d"].cuda_kernel,
        arrays,
        {"positions": 2, "field": 2, "result": 2},
        call,
        ["result"],
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)


QUADRATIC_CASES = [
    (1, (False,)),
    (1, (True,)),
    (2, (False, False)),
    (2, (True, False)),
    (2, (True, True)),
    (3, (False, False, False)),
    (3, (True, False, True)),
    (3, (True, True, True)),
]


def _quadratic_case(ndim, seed):
    """A grid of ``ndim`` axes, two components and markers partly outside it."""
    rng = np.random.default_rng(seed)
    shape, ncomp, n = (6, 5, 4)[:ndim], 2, 150
    lower = np.array([0.0, -1.0, 2.0])[:ndim]
    spacing = np.array([0.25, 0.5, 0.1])[:ndim]
    upper = lower + spacing * np.array(shape)
    points = rng.uniform(lower - 0.3, upper + 0.3, (n, ndim)).T.copy()  # (ndim, N)
    geometry = ", ".join(
        [repr(float(v)) for v in lower]
        + [repr(float(v)) for v in spacing]
        + [str(v) for v in shape]
    )
    return {
        "shape": shape,
        "ncomp": ncomp,
        "n": n,
        "lower": lower,
        "spacing": spacing,
        "points": points,
        "values": rng.normal(size=(ncomp, n)),
        "field": rng.normal(size=(*shape, ncomp)),
        "geometry": geometry,
    }


@pytest.mark.parametrize(("ndim", "periodic"), QUADRATIC_CASES)
def test_scatter_quadratic_matches_cpu_kernel(tmp_path, ndim, periodic):
    case = _quadratic_case(ndim, 8)
    case["values"][:, ::7] = 0.0
    case["points"][:, ::7] = np.nan  # dead markers may sit anywhere
    grid = BlockGrid.from_geometry(
        case["lower"], case["spacing"], case["shape"], periodic
    )
    expected = np.zeros_like(case["field"])
    with xp.kernels.use_host_kernel_implementation("numpy"):  # the CPU reference
        QUADRATIC.scatter_block(case["points"], case["values"], expected, grid)
    name = f"scatter_quadratic_{ndim}d"
    flags = ", ".join(str(int(p)) for p in periodic)
    call = (
        case["n"],
        f"{name}(positions_view, values_view, field_view, {case['geometry']}, {flags})",
    )
    (result,) = _run(
        tmp_path,
        KERNELS[name].cuda_kernel,
        {
            "positions": case["points"],
            "values": case["values"],
            "field": np.zeros_like(case["field"]),
        },
        {"positions": 2, "values": 2, "field": ndim + 1},
        call,
        ["field"],
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(  # all the charge is on the grid
        result.reshape(-1, case["ncomp"]).sum(axis=0), case["values"].sum(axis=1)
    )


@pytest.mark.parametrize(("ndim", "periodic"), QUADRATIC_CASES)
def test_gather_quadratic_matches_cpu_kernel(tmp_path, ndim, periodic):
    case = _quadratic_case(ndim, 9)
    grid = BlockGrid.from_geometry(
        case["lower"], case["spacing"], case["shape"], periodic
    )
    expected = np.zeros((case["ncomp"], case["n"]))  # (components, N)
    with xp.kernels.use_host_kernel_implementation("numpy"):
        QUADRATIC.gather_block(case["points"], case["field"], expected, grid)
    name = f"gather_quadratic_{ndim}d"
    flags = ", ".join(str(int(p)) for p in periodic)
    call = (
        case["n"],
        f"{name}(positions_view, field_view, result_view, {case['geometry']}, {flags})",
    )
    (result,) = _run(
        tmp_path,
        KERNELS[name].cuda_kernel,
        {
            "positions": case["points"],
            "field": case["field"],
            "result": np.zeros_like(expected),
        },
        {"positions": 2, "field": ndim + 1, "result": 2},
        call,
        ["result"],
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)
