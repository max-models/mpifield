"""Main module of the python package."""

import scope_profiler as sp

from mpifield._profiling import PROFILE_TAGS


@sp.profile("mpifield.main", tags=PROFILE_TAGS)
def main() -> None:
    """Print a greeting; the entry point of the ``mpifield`` command."""
    print("Hello, world")
