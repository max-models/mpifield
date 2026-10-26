"""Compile mpifield's Pyccel kernel modules once and reuse the builds across processes.

The cache is :class:`cunumpy.kernels.PyccelCache`: a kernel module is built with
``pyccel compile`` on its first use, keyed on its source, the Pyccel and Python
versions and the platform, and stored under ``MPIFIELD_KERNEL_CACHE`` (default
``~/.cache/mpifield/kernels``, or ``<tmp>/mpifield-kernels`` if that is not writable).
Later processes import the stored extension directly.

Kernels are built with OpenMP, so the loops marked ``#$ omp parallel for`` run on
all cores (limit them with ``OMP_NUM_THREADS``; under MPI set it to the cores per
rank). ``MPIFIELD_OPENMP=0`` builds them serial, except the kernels that call the OpenMP
runtime; the setting is part of the cache key.
"""

from __future__ import annotations

import cunumpy as xp

#: The on-disk build cache of mpifield's kernels.
KERNEL_CACHE = xp.kernels.PyccelCache("mpifield")

#: Return the compiled form of a kernel module, building it on first use
#: (:meth:`cunumpy.kernels.PyccelCache.compile`).
compile_cached = KERNEL_CACHE.compile
