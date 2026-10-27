# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `Field`, moved out of picnic (`picnic.field`): a field on the nodes of a uniform 1D, 2D or
  3D grid, split over MPI ranks with halo nodes (an mpiarray `DistributedArray` on a
  simgeometry `Grid`), with gather (`gather`, was `interpolate_field`) and scatter
  (`scatter`, was `scatter_linear`), halo exchange, arithmetic and ufuncs, reductions,
  smoothing filters, `evaluate` to xarray and `plot` with plasma-plots. With it `evaluate.py` and
  `prescribed.py` (`electric_field_from_potential`, `axial_potential`).
- `Weighting` (`mpifield.weighting`): a scatter and a gather kernel for each number of
  dimensions, and the halo nodes a split field needs for them. `Field.scatter` and
  `Field.gather` take one (`weighting=LINEAR` by default); a weighting of your own is a pair
  of functions or compiled kernels with the arguments of the module's kernel convention.
  `Weighting.scatter_block`/`gather_block` run one on any array, with
  `BlockGrid.from_geometry` for its grid. Built in: `LINEAR` (cloud-in-cell) and `QUADRATIC`
  (the quadratic B-spline, halo 2; a stencil node past a bounded edge falls on the edge node,
  so the deposit keeps all the charge).
- `mpifield.kernels`: the linear and quadratic kernels (`gather_linear_{1,2,3}d`,
  `scatter_linear_{1,2,3}d`, `gather_quadratic_{1,2,3}d`, `scatter_quadratic_{1,2,3}d`) with
  their Pyccel, NumPy, CUDA and Metal versions, the CUDA headers `linear_cell.cuh` and
  `quadratic_cell.cuh` and the kernel catalog; the Pyccel builds are cached under
  `MPIFIELD_KERNEL_CACHE`.
- Public names for what picnic's models used privately: `Field.local_with_halos` (the stored
  block the kernels work on), `Field.local`, `Field.block_grid()` (was `_stored_grid`) and
  `BlockGrid`, `Bounds`. `block_grid(inside_domain=True)` leaves out the halo nodes beyond a
  bounded edge of the domain; `BlockGrid.geometry` holds the kernels' grid arguments and
  `BlockGrid.index` the part of `local_with_halos` the grid describes.
- A scalar field broadcasts over the components of a vector field in arithmetic, on either
  side and in place (`E / E.magnitude()`).
- `Field.gradient()`: returns a new field with one component per axis, wraps around
  periodic axes, and does not depend on the split over ranks.
- `Field.set_inside(shape, value)`: sets the nodes inside a simgeometry shape, vectorised.
- `Field.export`/`Field.load` write and read NetCDF through xarray.
- Opt-in scope-profiler regions (tag `mpifield`) on the gather, the scatter, `smooth` and
  `gradient`.
- Tests serially and under MPI (CI: `mpiexec -n 2` and `-n 4`), the kernel tests (Pyccel
  against NumPy, the OpenMP deposit, the CUDA kernels compiled as C++ on the CPU, Metal), four
  executed tutorials and thirteen full examples.

### Changed (against picnic's `Field`)

- Removed: the B-spline members (`degree`, the knot vectors, `bspline_basis_functions`; no
  gather or scatter used them), `scatter_charge_density_linear` and `ChargeSource` (the PIC
  deposit stays in picnic), `compute_gradient`, `field_data_gradient` and
  `get_gradient_field_data` (an extra array for every scalar field; use `gradient()`),
  `set_field_in_object` (its node positions ignored the lower bound; use `set_inside`),
  `get_nearest_value`, `sync_ghost_points` (use `fill_halos`), `local_data_with_halos` (use
  `local_with_halos`), the pickle I/O.
- `copy`, `astype`, arithmetic, `dot` and `magnitude` keep the subclass of the field.
- Comparing with a non-field in `_check_field_compatibility` raises `TypeError` instead of
  returning `NotImplemented`.
- Removed: the static `Field.scatter_kernel_{1,2,3}d` and `Field.interpolate_field_{1,2,3}d`
  and the `scatter_kernel` and `interpolation_kernel` properties; use
  `LINEAR.scatter_block(positions, values, array, grid)` and
  `LINEAR.gather_block(positions, array, result, grid)`.
- `scatter_linear_1d` takes `periodic_x` like the other kernels.
- A split field with fewer halo nodes than the weighting needs raises in `scatter` and
  `gather` (one for linear weights), instead of losing the last cell's deposit.
- The scatter and gather of a split field leave out the halo nodes beyond a bounded edge of
  the domain, so markers there clamp to the edge as on an unsplit field.
- A number as the values of `scatter` goes to every component; values with another number of
  components than the field raise.
- The 3D scatter runs on the GPU for CuPy markers and field, as the 1D and 2D ones did.

### Fixed

- The 1D linear deposit wrapped around on a bounded axis: a marker past the upper end put
  part of its charge on the first node.
- The 1D linear gather did not wrap on a periodic axis: a marker in the last cell got the
  last node's value instead of interpolating to the first node.
