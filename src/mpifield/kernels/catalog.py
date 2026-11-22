"""mpifield's compiled kernels: one folder per kernel, with Pyccel, NumPy, CUDA and Metal versions.

Every kernel ``<name>`` lives in its own folder::

    mpifield/kernels/<name>/__init__.py        kernel = KERNEL_PACKAGE.from_folder(__name__)
    mpifield/kernels/<name>/<name>_pyccel.py   def <name>(...)               CPU, compiled with Pyccel
    mpifield/kernels/<name>/<name>_cuda.cu     __global__ void <name>(...)   GPU, compiled with NVRTC
    mpifield/kernels/<name>/<name>_numpy.py    def <name>(...)               CPU, when Pyccel is unavailable
    mpifield/kernels/<name>/<name>_metal.py    def <name>(...)               Apple-silicon GPU, float32

All versions take the same arguments, so a call site is the same on every device::

    from mpifield.kernels.gather_linear_2d import kernel as gather_linear_2d

    gather_linear_2d(positions, field, result, lower_x, lower_y, dx, dy, nx, ny, px, py)

runs the CUDA version when the arrays live on the GPU (CuPy) and the compiled Pyccel
version otherwise (``dispatch="arrays"``). The kernel object is a
`cunumpy.kernels.Kernel`, built in the folder's ``__init__.py`` by
``KERNEL_PACKAGE.from_folder`` (see `mpifield.kernels`). Both versions are
compiled on first use: importing this module needs neither Pyccel nor CuPy. CUDA
sources include mpifield's header relative to the source root:
``#include "mpifield/kernels/linear_cell.cuh"``.

The kernels are the linear (``*_linear_*``), quadratic (``*_quadratic_*``) and spline
(``*_spline_*``: B-splines of degree 0 to 3, chosen by their last argument ``degree``;
no Metal version yet) deposits and gathers of `mpifield.weighting`, and the edge gathers
(``gather_edges_*``) and path deposits (``deposit_paths_*``) of `mpifield.edges` (edge
fields ``num_nodes + (ndim,)``, paths ``(ndim, N)``, charges ``(N,)``; no Metal version
yet). The quadratic and spline ones (the latter with ``degree`` after the geometry), the
1D and 2D linear gathers and the 2D linear deposit take the arguments of the weighting
kernel convention; the linear 1D deposit takes one component (``(N,)`` markers, an
``(nx,)`` grid) and the linear 3D kernels the grid as rows ``(nx * ny * nz, ncomp)``.

Marker arrays are component-major: positions ``(ndim, N)``, gather results and
scatter values ``(ncomp, N)``. Grid fields are ``num_gridpoints + (ncomp,)``; the linear
3D kernels take them flattened to ``(nx * ny * nz, ncomp)`` rows.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import cunumpy as xp
from cunumpy.kernels import metal_available

__all__ = ["KERNELS", "KERNEL_NAMES", "METAL_KERNEL_NAMES", "metal_kernel"]

#: The kernels, by name.
KERNEL_NAMES = (
    "gather_linear_1d",
    "gather_linear_2d",
    "gather_linear_3d",
    "scatter_linear_1d",
    "scatter_linear_2d",
    "scatter_linear_3d",
    "gather_quadratic_1d",
    "gather_quadratic_2d",
    "gather_quadratic_3d",
    "scatter_quadratic_1d",
    "scatter_quadratic_2d",
    "scatter_quadratic_3d",
    "gather_spline_1d",
    "gather_spline_2d",
    "gather_spline_3d",
    "scatter_spline_1d",
    "scatter_spline_2d",
    "scatter_spline_3d",
    "gather_edges_1d",
    "gather_edges_2d",
    "gather_edges_3d",
    "deposit_paths_1d",
    "deposit_paths_2d",
    "deposit_paths_3d",
    "gather_edges_spline_1d",
    "gather_edges_spline_2d",
    "gather_edges_spline_3d",
    "deposit_paths_spline_1d",
    "deposit_paths_spline_2d",
    "deposit_paths_spline_3d",
    "gather_forms_spline_1d",
    "gather_forms_spline_2d",
    "gather_forms_spline_3d",
    "scatter_forms_spline_1d",
    "scatter_forms_spline_2d",
    "scatter_forms_spline_3d",
)

_PACKAGE = __name__.rpartition(".")[0]

#: Every kernel of mpifield, by name.
KERNELS = xp.kernels.KernelCatalog(
    {
        name: importlib.import_module(f"{_PACKAGE}.{name}").kernel
        for name in KERNEL_NAMES
    }
)

#: The kernels with a Metal version (``<name>/<name>_metal.py``): the linear and
#: quadratic node weightings', not yet the spline ones (``*_spline_*``) or the edge ones
#: (``gather_edges_*``, ``deposit_paths_*``).
METAL_KERNEL_NAMES = tuple(
    name
    for name in KERNEL_NAMES
    if not name.startswith(
        (
            "gather_edges",
            "deposit_paths",
            "gather_spline",
            "scatter_spline",
            "gather_forms",
            "scatter_forms",
        )
    )
)


def metal_kernel(name: str) -> Callable[..., None]:
    """Return the Metal version of kernel ``name``: a function with the Pyccel arguments.

    Needs MLX and a Metal GPU (``pip install 'mpifield[metal]'``); the arrays are NumPy
    arrays, computed in float32 and written back into the output arrays.

    Raises:
        KeyError: If the kernel has no Metal version.
        RuntimeError: If there is no usable Metal GPU.
    """
    if name not in METAL_KERNEL_NAMES:
        raise KeyError(f"{name!r} has no Metal version; have {METAL_KERNEL_NAMES}")
    if not metal_available():
        raise RuntimeError("Metal kernels need MLX and a Metal GPU (mpifield[metal])")
    module = importlib.import_module(f"{_PACKAGE}.{name}.{name}_metal")
    return getattr(module, name)
