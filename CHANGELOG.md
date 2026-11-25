# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- The threaded 2D and 3D linear deposits (`scatter_linear_2d`, `scatter_linear_3d`, more markers
  than `MIN_PARALLEL_MARKERS` on several OpenMP threads) skip a NaN or infinite position when
  they find the x-planes of each thread's markers. Such a marker (a zero value, which deposits
  nothing) gave an undefined plane, a wrong band and NaN nodes or a crash.

### Added

- `scatter_rows_3d`, a host-only kernel (C, Pyccel and NumPy versions): the one-component 3D
  linear deposit on contiguous rows of the positions. `LINEAR` takes it for NumPy arrays when
  its C version is built (cunumpy's C host kernels), and `scatter_linear_3d` otherwise: about
  1.25x faster on one core (14.5 to 11.6 ns per marker, 0.5M markers sorted by cell on a
  33 x 32 x 32 grid), the same result on one thread; on several threads each deposits into
  its own grid, summed at the end.

### Changed

- The 3D linear deposit (`scatter_linear_3d`, Pyccel) is about 30% faster on one core (18.2 to
  14.1 ns per marker for one component, 0.5M markers sorted by cell on a 33 x 32 x 32 grid),
  with the same results bit for bit: the cell of a marker in one form for bounded and periodic
  axes, and the one-component loop back on its own. The new inline blocks of
  `linear_cell_pyccel.py`, `wrapped_position`, `clamped_left_node` (with `last = num - 2 +
  periodic`) and `next_node`, give the results of `periodic_position`, `left_node` and
  `right_node` without a periodic flag in the loop, for kernels whose loops should vectorise.

### Added

- Clamped splines, the spaces of spline finite elements (GEMPIC, Struphy): a third boundary
  code per axis, 2 (`BlockGrid.from_geometry(..., clamped=...)`), where the knots are the
  nodes with the end ones repeated `p + 1` times and an array of degree p holds
  `num_nodes + p - 1` coefficients. The spline kernels (nodes, edges, faces, cells) take it
  through the shared blocks (`clamped_cell` in `spline_cell_pyccel.py`, `spline_cell.cuh`,
  `spline_cell.py`, Cox-de Boor); the staggered functions there are
  `D_i = p / (t_(i+p) - t_i) B_(i,p-1)`, so the path deposit conserves the charge exactly and
  faces and cells commute with the curl and divergence up to the walls, and no charge is
  lost at a wall. `splines(p, clamped=True)` (`CLAMPED_SPLINES`) returns the shapes whose
  weightings all take clamped axes; `Weighting.clamps` and `EdgeWeighting.clamps` say which
  do (the linear and quadratic kernels and `WHITNEY` refuse a clamped grid).
- `mpifield.bsplines`: the 1D clamped spline spaces with the same functions,
  `clamped_knots`, `num_functions`, `evaluate` (form 0 `B_i`, form 1 `D_i / h`),
  `quadrature`, `mass_matrix` and `incidence` (`G`, exact derivative), for spline finite
  element solvers without feectools or Struphy. Tested against `scipy.interpolate.BSpline`,
  also the folded and periodic weights inside the grid.
- The scatter kernels of the splines size their thread buffers from the field, so arrays
  larger than the node grid (clamped coefficients) deposit correctly on many threads.

- B-spline shapes by degree (`mpifield.splines`): `Splines`, the node weighting of a degree
  and the edge weighting that matches it (whose path deposit conserves its charge), and
  `splines(p)` returning the built-in one (`SPLINES`: degree 1 `(LINEAR, WHITNEY)`, 2
  `(QUADRATIC, QUADRATIC_EDGES)`, 3 `(CUBIC, CUBIC_EDGES)`). `Weighting.degree` and `EdgeWeighting.degree` give
  the degree of a built-in weighting (`None` for one of your own).
- Faces and cells of the spline sequence: `Splines.faces` (2-forms in 3D, degree p along the
  normal and p - 1 across) and `Splines.cells` (the top form, p - 1 along every axis), each a
  `Weighting` on values that are integrals over the faces or cells. At bounded walls the
  `B_p` factors fold and the staggered `B_{p-1}` ones drop the k-cells past the wall, so the
  gathered fields commute with the discrete derivatives: the faces of `C e` gather the curl
  of the edge field, the cells of `D f` the divergence of the face field. Kernels
  `gather_forms_spline_{1,2,3}d` and `scatter_forms_spline_{1,2,3}d` with the degree and the
  form as their last arguments, in Pyccel (OpenMP), NumPy (`kernels/spline_forms.py`) and
  CUDA.
- `QUADRATIC_EDGES` and `CUBIC_EDGES`: the edge weightings of degree 2 and 3, paired with
  `QUADRATIC` and `CUBIC` in `splines(p)`. The function of an edge along axis a is
  `B_{p-1}(r_a - k_a - 1/2) prod_{b != a} B_p(r_b - k_b)`, so the path deposit conserves the
  charge of the node weighting of the same degree exactly (`divergence_block`), also at
  bounded walls (paths clamped, edges past a wall dropped) and periodic seams; each path
  piece between knot planes is integrated exactly by Gauss-Legendre quadrature, and the work
  `j . e` is the integral of the gathered field along the paths. Kernels
  `gather_edges_spline_{1,2,3}d` and `deposit_paths_spline_{1,2,3}d` with the degree (1 to 3)
  as their last argument, in Pyccel (the gather with OpenMP, the path deposit serial), NumPy
  (`kernels/spline_edges.py`, the reference) and CUDA, sharing `spline_edges_pyccel.py` and
  `spline_edges.cuh`; degree 1 reproduces `WHITNEY`.
- `CUBIC`, the cubic B-spline weighting (four nodes per axis, halo 2), on new kernels whose
  degree (0 to 3) is an argument: `scatter_spline_{1,2,3}d` and `gather_spline_{1,2,3}d` in
  Pyccel (OpenMP), NumPy and CUDA (no Metal yet), sharing the stencil block
  `kernels/spline_cell_pyccel.py`, `kernels/spline_cell.cuh` and `kernels/spline_cell.py`.
  Degree 1 and 2 reproduce `LINEAR` and `QUADRATIC`; degree 0 is the nearest grid point.
  Example 13 compares the cubic noise with the others.

- Edge fields (`mpifield.edges`): `WHITNEY`, an `EdgeWeighting` pairing the lowest-order
  edge (Whitney) gather (`gather_block`) with the deposit of the current of charges moving
  along straight paths (`deposit_paths_block`) on a block, in 1D, 2D and 3D, with the block
  incidence `gradient_block` and its transpose `divergence_block`. An edge field is
  `num_nodes + (ndim,)` line integrals. `divergence_block` of the deposit is `LINEAR`'s charge
  at the ends minus at the starts, to round-off, also across periodic seams and with paths
  clamped at bounded edges. Kernels `gather_edges_{1,2,3}d` and `deposit_paths_{1,2,3}d` in
  Pyccel (OpenMP), NumPy and CUDA (no Metal yet); `kernels/whitney_cell_pyccel.py` and
  `kernels/whitney_cell.cuh` hold the edge weights for other packages' kernels (picnic's
  implicit movers). Example 15 (Edge fields and currents).

- `mpifield/kernels/linear_cell_pyccel.py`: the linear cell for Pyccel kernels, `@inline`
  `periodic_position`, `left_node`, `right_node` and `cell_fraction` without local variables
  (safe inside `#$ omp parallel for`), the Pyccel version of `linear_cell.cuh` and
  `linear_cell.py`. Pyccel kernels of any package import them and cunumpy's kernel cache
  builds them with the kernel (needs cunumpy with dependency builds, cunumpy#138). The linear
  gathers and scatters use them: the same results bit for bit and the same speed.

### Changed

- The kernels are built with cunumpy's `KernelPackage` (`mpifield.kernels.KERNEL_PACKAGE`):
  `kernels/_build.py`, `kernels/_metal.py` and `kernels/compile.py` are gone, and the build
  cache is still `MPIFIELD_KERNEL_CACHE`. `shared_memory_fits` is
  `xp.cuda.shared_memory_fits`; the catalog no longer exports it.
- Needs `cunumpy>=0.7.2`, the first release with `KernelPackage` (was `>=0.7`, with which
  `import mpifield` failed on cunumpy 0.7.0 and 0.7.1).

### Added

- A live demo in the docs (Getting started, Live demo): markers deposited onto a 1D grid with
  `Field.deposit_density`, run in the browser with Pyodide on the NumPy kernels, with sliders
  for the markers, the nodes, the weighting and the smoothing
  (`docs/src/components/mpifield_demo.py`, starlight-theme's `PyodideDemo`). A Pyodide
  workflow checks that every module imports in the browser and that the demo runs.
- mpifield installs in the browser (Pyodide): Pyccel, which has no wheel there, is not a
  dependency on `sys_platform == 'emscripten'`, and the kernels run their NumPy versions
  (cunumpy 0.8.0 imports a Pyccel source only when its version is used). Needs
  `cunumpy>=0.8.0`.
- `Field.deposit_density(positions, values, weighting=LINEAR)`: adds the deposit of the
  marker values divided by the node volumes, with the halo exchange on a split field; with
  the products `w`, `w v`, `w v^2`, ... as components it gives the velocity moments.
  Example 14 derives the flow, temperature and heat flux from them.
- `Field.node_volumes()`: the volume each node collects in a deposit, half a cell on a bounded
  edge. `Field.on_grid` takes `num_ghost_points` (default 1).
- Method cards (methodcard) on the numerics: the linear and quadratic shape functions
  (`linear_cell`, `quadratic_cell`), the deposit and the gather (`Weighting.scatter_block`,
  `Weighting.gather_block`) and their adjointness (`Weighting`), the block grid and the halo
  exchanges, the gradient, the smoothing filters, the prescribed field and the evaluation,
  with their assumptions and references. `test_method_cards.py` checks them: 19 SymPy rules
  (the shape functions, partition of unity and first moment, edge folds and seams through the
  compiled kernels; the operators), the second order of the two gathers, the gradient and
  `electric_field_from_potential`, and the cards of the docs against those of the code. Adds
  the `methodcard` dependency (`methodcard[symbolic]` in the test extra).
- A Numerics section in the docs (header and sidebar): an overview
  (`docs/src/content/docs/numerics/index.mdx`) and one page per module
  (starlight-theme's `methodcard: { pages: true }`, theme 0.7.4); each method's API page shows
  its card.
- The cross references in the docstrings render on the API pages: code spans instead of
  `:class:`, `:meth:` and `:func:` roles.
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
