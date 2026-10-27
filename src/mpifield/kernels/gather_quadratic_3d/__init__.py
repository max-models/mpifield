"""The gather_quadratic_3d kernel: Pyccel, NumPy and CUDA versions with the same arguments.

``kernel(...)`` runs the CUDA version when the arrays are CuPy arrays, and
otherwise the Pyccel version, or the NumPy version if Pyccel is unavailable
(see ``cunumpy.kernels.set_host_kernel_implementation`` to choose).
"""

from mpifield.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
