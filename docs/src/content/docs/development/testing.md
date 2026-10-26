---
title: Testing
description: The test suite, serially and under MPI, and what each part checks.
---

```bash
make test                                                              # serially
mpiexec -n 2 .venv/bin/python -m pytest --with-mpi src/mpifield/tests  # under MPI
```

| Directory | Checks |
| --- | --- |
| `src/mpifield/tests/unit` | `Field` on one process: construction, data access, gather and scatter, arithmetic, gradients, smoothing, I/O, evaluation and plotting |
| `src/mpifield/tests/parallel` | that results do not depend on the split: deposits with halo exchanges, smoothing over halos, gradients, `set_inside`, I/O, run on any number of ranks |
| `src/mpifield/tests/kernels` | each kernel folder has all versions with one signature; Pyccel against NumPy; the parallel (OpenMP) deposit; the CUDA versions compiled as C++ on the CPU; the Metal versions (on Apple silicon with MLX) |

Tests marked `@pytest.mark.mpi(min_size=2)` run only with `--with-mpi` (pytest-mpi). CI runs
the suite serially on Python 3.10 to 3.13, and under `mpiexec -n 2` and `-n 4` with Open MPI.

A test directory must not be called `mpi`: pytest-mpi skips every test whose keywords contain
"mpi" without `--with-mpi`, and the directory name is one of them.
