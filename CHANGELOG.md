# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- The kernels are built with cunumpy's `KernelPackage` (`mpifield.kernels.KERNEL_PACKAGE`):
  `kernels/_build.py`, `kernels/_metal.py` and `kernels/compile.py` are gone, and the build
  cache is still `MPIFIELD_KERNEL_CACHE`. `shared_memory_fits` is
  `xp.cuda.shared_memory_fits`; the catalog no longer exports it.

### Added

- `Field`, moved out of picnic (`picnic.field`): a field on the nodes of a uniform 1D, 2D or
  3D grid, split over MPI ranks with halo nodes (an mpiarray `DistributedArray` on a
  simgeometry `Grid`), with linear gather (`interpolate_field`) and scatter
  (`scatter_linear`), halo exchange, arithmetic and ufuncs, reductions, smoothing filters,
  `evaluate` to xarray and `plot` with plasma-plots. With it `evaluate.py` and
  `prescribed.py` (`electric_field_from_potential`, `axial_potential`).
- `mpifield.kernels`: the six linear kernels (`gather_linear_{1,2,3}d`,
  `scatter_linear_{1,2,3}d`) with their Pyccel, NumPy, CUDA and Metal versions, the CUDA
  header `linear_cell.cuh` and the kernel catalog; the Pyccel builds are cached under
  `MPIFIELD_KERNEL_CACHE`.
- Public names for what picnic's models used privately: `Field.local_with_halos` (the stored
  block the kernels work on), `Field.local`, `Field.block_grid()` (was `_stored_grid`) and
  `BlockGrid`, `Bounds`.
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
  executed tutorials and twelve full examples.

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
