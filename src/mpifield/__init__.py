"""Template python package."""

from importlib.metadata import PackageNotFoundError, version

import scope_profiler as sp

from mpifield._profiling import PROFILE_TAGS

try:
    __version__ = version("mpifield")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "0.0.0+unknown"

# The decorated functions pick the tag up, so registering after the imports is in time.
sp.register_opt_in_tag(*PROFILE_TAGS)
