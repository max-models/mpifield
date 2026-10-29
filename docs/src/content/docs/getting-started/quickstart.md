---
title: Quickstart
description: A field, a deposit, a gather and a plot.
---

First, ensure that `mpifield` is [installed](/mpifield/getting-started/installation/), with the
`plot` extra for the plots.

## A field

A `Field` is a uniform grid (a simgeometry `Grid`) with values on its nodes:

```python
import numpy as np
from mpifield import Field

phi = Field(
    ndim=2, lower_bound=(0.0, 0.0), upper_bound=(0.1, 0.05), num_gridpoints=(101, 51)
)
x, y = np.meshgrid(*phi.get_grid_points(), indexing="ij")
phi.set_field_data(1000.0 * np.exp(-((x - 0.05) ** 2 + (y - 0.025) ** 2) / 1e-4))
E = -phi.gradient()  # two components, V/m
```

## Markers

Positions are component-major, `(ndim, N)`; values `(ncomp, N)`.

```python
positions = np.random.default_rng(0).random((2, 10_000)) * [[0.1], [0.05]]

field_at_markers = E.gather(positions)  # (2, N), the linear gather

density = Field(
    ndim=2, lower_bound=(0.0, 0.0), upper_bound=(0.1, 0.05), num_gridpoints=(101, 51)
)
density.scatter(positions, np.ones((1, 10_000)))  # the cloud-in-cell deposit
print(density.sum())  # 10000.0: the deposit conserves the total
```

## Looking at it

`evaluate` gives an `xarray.DataArray` of the nodes, a line or a plane; `plot` draws it with
[plasma-plots](https://github.com/max-models/plasma-plots):

```python
profile = phi.evaluate(y=0.025)  # a line along x
phi.plot(name="potential", units="V")  # a map
E.evaluate().plasma.plot.vector(x="x", y="y", stride=5)
```

## Under MPI

The same code splits the grid over the ranks with `comm` and `domain_decomposition`; deposits
are completed with `exchange_halos`, and halos are filled for gathers with `fill_halos`:

```python
from maybempi import MPI

density = Field(
    ndim=2,
    lower_bound=(0.0, 0.0),
    upper_bound=(0.1, 0.05),
    num_gridpoints=(101, 51),
    comm=MPI.COMM_WORLD,
    domain_decomposition=[True, True],
    num_ghost_points=1,
)
```

See the [MPI and halos](/mpifield/examples/mpi-and-halos/) example, and the
[example scripts](/mpifield/example-scripts/) for a gallery of complete scripts. The
[API reference](/mpifield/api/mpifield/) documents every class and function.

## Profiling

mpifield's coarse operations (`mpifield.Field.gather`, `scatter`, `smooth`,
`gradient`) are [scope-profiler](https://github.com/max-models/scope-profiler) regions with
the opt-in tag `mpifield`: off by default, they cost a fraction of a microsecond per call.
Turn them on with `scope_profiler.setup(enable_tags=["mpifield"])`,
`SCOPE_PROFILER_ENABLE_TAGS=mpifield` or `scope-profiler run --enable-tag mpifield`.
