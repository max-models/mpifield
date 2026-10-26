"""Fields on uniform grids split over MPI ranks, with compiled linear gather and scatter.

:class:`Field` holds a field on the nodes of a uniform 1D, 2D or 3D grid as an
:class:`mpiarray.DistributedArray`; it interpolates the field at marker positions
(:meth:`Field.interpolate_field`) and deposits marker values onto the nodes
(:meth:`Field.scatter_linear`) through the compiled kernels of
:mod:`mpifield.kernels`, on the CPU (Pyccel, NumPy) or the GPU (CUDA, Metal).
"""

from importlib.metadata import PackageNotFoundError, version

import scope_profiler as sp

from mpifield._profiling import PROFILE_TAGS
from mpifield._types import Bounds
from mpifield.field import BlockGrid, Field
from mpifield.prescribed import axial_potential, electric_field_from_potential

try:
    __version__ = version("mpifield")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "0.0.0+unknown"

# The decorated functions pick the tag up, so registering after the imports is in time.
sp.register_opt_in_tag(*PROFILE_TAGS)

__all__ = [
    "BlockGrid",
    "Bounds",
    "Field",
    "__version__",
    "axial_potential",
    "electric_field_from_potential",
]
