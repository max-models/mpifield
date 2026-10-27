# mpifield


<!-- README.md is generated from README.qmd: edit the .qmd and run `make readme`. -->

[![Tests](https://github.com/max-models/mpifield/actions/workflows/test_pytest.yml/badge.svg)](https://github.com/max-models/mpifield/actions/workflows/test_pytest.yml)
[![Static
analysis](https://github.com/max-models/mpifield/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/max-models/mpifield/actions/workflows/static_analysis.yml)
[![Docs](https://github.com/max-models/mpifield/actions/workflows/docs.yml/badge.svg)](https://max-models.github.io/mpifield/)
[![codecov](https://codecov.io/gh/max-models/mpifield/branch/main/graph/badge.svg)](https://codecov.io/gh/max-models/mpifield)
[![PyPI](https://img.shields.io/pypi/v/mpifield.png)](https://pypi.org/project/mpifield/)
[![Python](https://img.shields.io/pypi/pyversions/mpifield.png)](https://pypi.org/project/mpifield/)

Fields on uniform grids split over MPI ranks, with compiled gather and
scatter for particle codes. A `Field` holds the values on the nodes of a
1D, 2D or 3D grid (one or several components per node) as an
[mpiarray](https://github.com/max-models/mpiarray) distributed array
with halo nodes; it interpolates the field at marker positions and
deposits marker values onto the nodes with linear (cloud-in-cell) or
quadratic B-spline weights, or a weighting of your own, through kernels
compiled with Pyccel (CPU, OpenMP), CUDA (CuPy arrays) or Metal
(Apple-silicon GPUs). It was moved out of
[picnic](https://github.com/max-models/picnic), so other particle codes
can use it without the rest of picnic.

Documentation: <https://max-models.github.io/mpifield/>

``` python
import numpy as np

from mpifield import QUADRATIC, Field

kwargs = {"ndim": 2, "lower_bound": (0.0, 0.0), "upper_bound": (0.1, 0.05)}
phi = Field(num_gridpoints=(101, 51), **kwargs)
x, y = np.meshgrid(*phi.get_grid_points(), indexing="ij")
phi.set_field_data(1000.0 * np.exp(-((x - 0.05) ** 2 + (y - 0.025) ** 2) / 1e-4))

E = -phi.gradient()  # a field with one component per axis
positions = np.random.default_rng(0).random((2, 10_000)) * [[0.1], [0.05]]
E_at_markers = E.gather(positions)  # (2, N)

density = Field(num_gridpoints=(101, 51), **kwargs)
density.scatter(positions, np.ones((1, 10_000)))  # cloud-in-cell deposit
smooth = Field(num_gridpoints=(101, 51), **kwargs)
smooth.scatter(positions, np.ones((1, 10_000)), QUADRATIC)  # quadratic B-spline
phi.plot(name="potential", units="V")  # through plasma-plots
```

## What is in it

- `Field`: construction (`Field(...)`, `from_array`, `on_grid` for a
  simgeometry grid), data access (`set_field_data`, `local`,
  `local_with_halos`, `to_numpy`), halo exchange (`fill_halos`,
  `exchange_halos`), gather and scatter (`gather`, `scatter`),
  arithmetic and ufuncs (a scalar field broadcasts over a vector field’s
  components), `gradient`, `cross`, `dot`, `magnitude`, global
  reductions, smoothing filters, `set_inside` for simgeometry shapes,
  `evaluate` to xarray, `plot` with plasma-plots, and NetCDF
  `export`/`load`.
- `Weighting`: a deposit and its gather for each number of dimensions,
  and the halo nodes they need. `LINEAR` (the default) and `QUADRATIC`
  are built in; a weighting of your own is a pair of functions or
  compiled kernels with the arguments of `mpifield.weighting`’s kernel
  convention. `Weighting.scatter_block` and `gather_block` run one on
  plain arrays, e.g. one patch of a refined mesh.
- `mpifield.kernels`: the twelve kernels of the two weightings
  (`gather_linear_{1,2,3}d`, `scatter_linear_{1,2,3}d`,
  `gather_quadratic_{1,2,3}d`, `scatter_quadratic_{1,2,3}d`), each with
  Pyccel, NumPy, CUDA and Metal versions.
- `electric_field_from_potential` and `axial_potential`: fields from
  prescribed potentials.

## Install

``` bash
pip install mpifield
pip install "mpifield[mpi,plot]"   # mpi4py for MPI runs, matplotlib and plasma-plots for plots
```

| Extra | Adds |
|----|----|
| `mpi` | `mpi4py`, for runs under `mpiexec`/`srun` (serial runs use maybempi’s stand-in) |
| `plot` | matplotlib and plasma-plots, for `Field.plot` |
| `gpu` | CuPy, for the CUDA kernels |
| `metal` | MLX, for the Metal kernels on Apple silicon |
| `test`, `docs`, `dev` | the test runner, the documentation tooling, the linters |

The CPU kernels are compiled with Pyccel on first use and cached under
`~/.cache/mpifield/kernels` (`MPIFIELD_KERNEL_CACHE`);
`MPIFIELD_OPENMP=0` builds them serial.
`cunumpy.kernels.set_host_kernel_implementation("numpy")` runs the NumPy
versions instead, e.g. on a machine without a C compiler.

## Development

With [uv](https://docs.astral.sh/uv/):

``` bash
make install    # uv sync --extra dev, plus the pre-commit hooks
make lint       # ruff check, ruff format --check, pyright
make test       # pytest with coverage
mpiexec -n 2 .venv/bin/python -m pytest --with-mpi src/mpifield/tests   # under MPI
```

The tests in `src/mpifield/tests/parallel` check that the results do not
depend on the number of ranks; CI runs the whole suite serially and
under `mpiexec -n 2` and `-n 4`.

## Build docs

The documentation in `docs/` is an Astro + Starlight site: hand-written
pages, the notebooks in `tutorials/` executed and published as pages,
the scripts in `examples/` published with their plots, and the API
reference generated from the docstrings. It needs Node 22 or newer.

``` bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-examples    # run examples/*.py and publish their plots
make docs-dev         # live preview at http://localhost:4321/mpifield/
make docs-build       # the static site in docs/dist
```

`README.md` is rendered from `README.qmd` with
[Quarto](https://quarto.org/): `make readme`.

## Releases

Before merging a release to `main`, update the version in
`pyproject.toml` and add the release notes to `CHANGELOG.md`. The push
to `main` creates a GitHub release with a `vX.Y.Z` tag and publishes the
package to PyPI with trusted publishing (OIDC); see the [publishing
guide](https://max-models.github.io/mpifield/development/publishing/).
