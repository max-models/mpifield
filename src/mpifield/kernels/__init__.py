"""mpifield's compiled kernels, one folder per kernel (see `mpifield.kernels.catalog`).

Every kernel folder builds its kernel with `KERNEL_PACKAGE`::

    kernel = KERNEL_PACKAGE.from_folder(__name__)

with the defaults of `cunumpy.kernels.KernelPackage`: the Pyccel version is
compiled on its first host call and kept under ``MPIFIELD_KERNEL_CACHE`` (default
``~/.cache/mpifield/kernels``; ``MPIFIELD_OPENMP=0`` builds serial), the CUDA version runs
for CuPy arguments and the host version for NumPy ones, one CUDA thread per entry of
the last axis of the first array. Importing a kernel needs neither Pyccel nor CuPy.

The Metal versions (``<name>/<name>_metal.py``) wrap a kernel built by
``KERNEL_PACKAGE.metal_kernel`` in a function with the arguments of the Pyccel
version. The Apple GPU has no float64, so they compute in float32: expect a relative
error of about 1e-5 against the CPU versions. Import one only where
`cunumpy.kernels.metal_available` is true.

In the browser (Pyodide) Pyccel cannot be installed (it needs a compiler), so the
kernels run their NumPy versions; ``_add_openmp_stand_in`` lets the Pyccel sources,
which import Pyccel's OpenMP module, be imported there.
"""

import importlib.util
import sys
import types
from collections.abc import MutableMapping

import cunumpy as xp


def _add_openmp_stand_in(modules: MutableMapping[str, types.ModuleType]) -> None:
    """Add a serial ``pyccel.stdlib.internal.openmp`` (one thread) to ``modules``.

    For a Python without Pyccel, where the Pyccel sources would fail to import;
    the stand-in has no package metadata, so no kernel tries to compile with it.
    """
    openmp = types.ModuleType("pyccel.stdlib.internal.openmp")
    openmp.omp_get_max_threads = lambda: 1  # type: ignore[attr-defined]
    openmp.omp_get_thread_num = lambda: 0  # type: ignore[attr-defined]
    child = openmp
    for name in ("pyccel.stdlib.internal", "pyccel.stdlib", "pyccel"):
        package = types.ModuleType(name)
        package.__path__ = []  # a package, so that its submodules import
        setattr(package, child.__name__.rpartition(".")[2], child)
        modules[child.__name__] = child
        child = package
    modules[child.__name__] = child


if sys.platform == "emscripten" and importlib.util.find_spec("pyccel") is None:
    _add_openmp_stand_in(sys.modules)

#: The build cache and launch options of mpifield's kernels.
KERNEL_PACKAGE = xp.kernels.KernelPackage("mpifield")
