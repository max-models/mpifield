"""The scope-profiler tag of mpifield's profiling regions.

The regions are opt-in: :mod:`mpifield` registers the tag on import, so they
record nothing unless the application enables it, e.g. with
``scope_profiler.setup(enable_tags=["mpifield"])``.
"""

#: Tags of every mpifield region; the first is the opt-in tag.
PROFILE_TAGS = ("mpifield",)
