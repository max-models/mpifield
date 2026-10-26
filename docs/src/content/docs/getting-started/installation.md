---
title: Installation
description: Install mpifield and its optional extras.
---

mpifield needs Python 3.10 or newer. Install it from PyPI:

```bash
pip install mpifield
```

or, for development, from a checkout with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/max-models/mpifield.git
cd mpifield
make install    # uv sync --extra dev, plus the pre-commit hooks
```

## Optional extras

| Extra   | Installs | For |
| ------- | -------- | --- |
| `mpi`   | mpi4py | runs under `mpiexec` or `srun`; serial runs use maybempi's stand-in and never start MPI |
| `plot`  | matplotlib, plasma-plots | `Field.plot` |
| `gpu`   | CuPy | the CUDA kernels, for fields and markers on the GPU |
| `metal` | MLX | the Metal kernels on Apple-silicon Macs |
| `test`  | pytest, pytest-mpi, coverage | the tests |
| `docs`  | the notebook runner, griffe | this site |
| `dev`   | ruff, pyright, pre-commit, plus `test` and `docs` | development |

```bash
pip install "mpifield[mpi,plot]"
```

## The compiled kernels

The CPU kernels are Python files compiled with [Pyccel](https://github.com/pyccel/pyccel) the
first time they run, with OpenMP, and cached on disk:

| Variable | Default | Effect |
| --- | --- | --- |
| `MPIFIELD_KERNEL_CACHE` | `~/.cache/mpifield/kernels` | where the compiled kernels are kept |
| `MPIFIELD_OPENMP` | `1` | `0` builds them without OpenMP |
| `OMP_NUM_THREADS` | all cores | threads per process; under MPI, the cores per rank |

Compiling needs a C compiler. Without one, run the NumPy versions:
`cunumpy.kernels.set_host_kernel_implementation("numpy")`.
