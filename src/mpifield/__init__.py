"""Fields on uniform grids split over MPI ranks, with compiled gather and scatter.

`Field` holds a field on the nodes of a uniform 1D, 2D or 3D grid as an
`mpiarray.DistributedArray`; it interpolates the field at marker positions
(`Field.gather`) and deposits marker values onto the nodes
(`Field.scatter`) with a `Weighting`: `LINEAR` (the default),
`QUADRATIC` or one of your own. The built-in ones run compiled kernels of
`mpifield.kernels`, on the CPU (Pyccel, NumPy) or the GPU (CUDA, Metal).
"""

from importlib.metadata import PackageNotFoundError, version

import scope_profiler as sp

from mpifield._profiling import PROFILE_TAGS
from mpifield._types import BlockGrid, Bounds
from mpifield.field import Field
from mpifield.prescribed import axial_potential, electric_field_from_potential
from mpifield.weighting import LINEAR, QUADRATIC, Weighting

try:
    __version__ = version("mpifield")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "0.0.0+unknown"

# The decorated functions pick the tag up, so registering after the imports is in time.
sp.register_opt_in_tag(*PROFILE_TAGS)

__all__ = [
    "LINEAR",
    "QUADRATIC",
    "BlockGrid",
    "Bounds",
    "Field",
    "Weighting",
    "__version__",
    "axial_potential",
    "electric_field_from_potential",
]
