"""Fields on uniform grids split over MPI ranks, with compiled gather and scatter.

`Field` holds a field on the nodes of a uniform 1D, 2D or 3D grid as an
`mpiarray.DistributedArray`; it interpolates the field at marker positions
(`Field.gather`) and deposits marker values onto the nodes
(`Field.scatter`) with a `Weighting`: `LINEAR` (the default),
`QUADRATIC`, `CUBIC` or one of your own. The built-in ones run compiled kernels of
`mpifield.kernels`, on the CPU (Pyccel, NumPy) or the GPU (CUDA, Metal).
`WHITNEY` gathers the field of an edge field (line integrals along the edges) and
deposits the current of charges moving along paths onto the edges, conserving
`LINEAR`'s charge exactly. `splines` returns the shape of a B-spline degree: the
node weighting and the edge weighting that matches it (`Splines`), which the codes
take instead of pairing weightings themselves.
"""

from importlib.metadata import PackageNotFoundError, version

import scope_profiler as sp

from mpifield._profiling import PROFILE_TAGS
from mpifield._types import BlockGrid, Bounds
from mpifield.edges import CUBIC_EDGES, QUADRATIC_EDGES, WHITNEY, EdgeWeighting
from mpifield.field import Field
from mpifield.prescribed import axial_potential, electric_field_from_potential
from mpifield.splines import SPLINES, Splines, splines
from mpifield.weighting import CUBIC, LINEAR, QUADRATIC, Weighting

try:
    __version__ = version("mpifield")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "0.0.0+unknown"

# The decorated functions pick the tag up, so registering after the imports is in time.
sp.register_opt_in_tag(*PROFILE_TAGS)

__all__ = [
    "CUBIC",
    "CUBIC_EDGES",
    "LINEAR",
    "QUADRATIC",
    "QUADRATIC_EDGES",
    "SPLINES",
    "WHITNEY",
    "BlockGrid",
    "Bounds",
    "EdgeWeighting",
    "Field",
    "Splines",
    "Weighting",
    "__version__",
    "axial_potential",
    "electric_field_from_potential",
    "splines",
]
