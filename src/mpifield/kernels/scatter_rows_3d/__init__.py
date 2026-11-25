"""The scatter_rows_3d kernel: C, Pyccel and NumPy versions with the same arguments.

A host-only kernel (no CUDA version, so not in the catalog): the one-component 3D linear
deposit on contiguous rows, which `mpifield.LINEAR` takes for NumPy arrays when its C
version is built (``scatter_linear_3d`` otherwise). The Pyccel and NumPy versions are its
references; the C one is preferred where a C compiler is available.
"""

from mpifield.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
