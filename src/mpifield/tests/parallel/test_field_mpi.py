"""Tests for field MPI."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI

from mpifield import LINEAR, QUADRATIC, Field

# mpirun -n 2 python src/mpifield/tests/parallel/test_field_mpi.py
# mpirun -n 2 pytest src/mpifield/tests/parallel/test_field_mpi.py --with-mpi


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("num_ghost_points", [0, 1, 2, 3, 4])
def test_initialize_field_mpi(num_ghost_points: int, verbose: bool = False) -> None:
    """Verify MPI field initialization."""
    comm = MPI.COMM_WORLD
    field1 = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(1.0,),
        # every rank needs at least ``num_ghost_points`` cells
        num_gridpoints=max(10, 4 * comm.Get_size()),
        num_components=1,
        num_ghost_points=num_ghost_points,
        comm=comm,
    )

    field1.set_value(1, 1.0, 0)
    field1.set_value(6, 2.0, 0)
    # print(field1.field_data[:, 0], field1.mpi_rank, field1.mpi_size, field1.proc_sizes, field1.num_gridpoints_local, field1.proc_index_bounds, field1.get_value(1,0), field1.get_value(6,0))
    # print(f"{field1.get_value(0, 0) = }")
    # print(f"{field1.mpi_rank = } {field1.field_data.local_with_halos = }")
    # print(f"{field1.get_value(0, 0)  = }")
    # print(f"{field1.get_value(1, 0)  = }")
    # print(f"{field1.get_value(6, 0)  = }")

    assert field1.get_value(0, 0) == 0.0
    assert field1.get_value(1, 0) == 1.0
    assert field1.get_value(6, 0) == 2.0


@pytest.mark.mpi(min_size=2)
def test_smoothing_matches_serial_reference_for_decomposed_field() -> None:
    """Smoothing on decomposed fields matches non-decomposed reference output."""
    comm = MPI.COMM_WORLD
    global_data = xp.zeros((32,), dtype=float)
    global_data[5] = 1.0
    global_data[21] = -0.5

    reference = Field.from_array(
        global_data,
        lower_bound=(0.0,),
        upper_bound=(1.0,),
        comm=comm,
        domain_decomposition=[False],
        periodic=True,
    )
    reference.smooth_gaussian(passes=2, in_place=True)

    decomposed = Field.from_array(
        global_data,
        lower_bound=(0.0,),
        upper_bound=(1.0,),
        comm=comm,
        domain_decomposition=[True],
        periodic=True,
    )
    decomposed.smooth_gaussian(passes=2, in_place=True)

    assert xp.allclose(decomposed.to_ndarray(), reference.to_ndarray())


# TODO: Implement ghost regions for Field
def test_ghost_regions(verbose: bool = False) -> None:
    """Verify field ghost-region exchange."""
    comm = MPI.COMM_WORLD
    field1 = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(1.0,),
        num_gridpoints=10,
        num_components=1,
        num_ghost_points=1,
        comm=comm,
        domain_decomposition=[True],
    )

    field1.set_value(1, 1.0, 0)
    field1.set_value(6, 2.0, 0)
    # print(
    #     field1.field_data[:, 0],
    #     field1.mpi_rank,
    #     field1.mpi_size,
    #     field1.proc_sizes,
    #     field1.num_gridpoints_local,
    #     field1.proc_index_bounds,
    #     field1.get_value(1, 0),
    #     field1.get_value(6, 0),
    # )
    print(f"{field1.field_data.local_with_halos =}")
    print(f"{field1.get_value(index=0, component=0) =}")
    # assert field1.get_value(index=0, component=0) == 0.0, f"{field1.get_value(index=0, component=0) = } but should be 0.0"
    # assert field1.get_value(1, 0) == 1.0
    # assert field1.get_value(6, 0) == 2.0

    # print(field1)
    # field1.sync_ghost_points()

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    # periodic = True
    # num_ghost = 2
    # N_local = 10 + 2 * num_ghost  # 10 interior points + 2 ghost points
    # Ny = 2

    # # Fill array with rank + 1
    # x = xp.full((N_local, Ny), fill_value=rank + 1, dtype=float)

    # # Determine neighbors: use MPI.PROC_NULL for “no neighbor” (edge ranks)
    # # This should be adapted per dimension
    # if periodic:
    #     left = (rank - 1) % size
    #     right = (rank + 1) % size
    # else:
    #     left = rank - 1 if rank > 0 else MPI.PROC_NULL
    #     right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    # # Allocate buffers
    # sendbuf = xp.empty((num_ghost, Ny), dtype=float)
    # recvbuf = xp.empty((num_ghost, Ny), dtype=float)
    # TAG = 0

    # # Stagger by parity:
    # if rank % 2 == 0:
    #     # even ranks do left → right
    #     if left != MPI.PROC_NULL:
    #         # print(f"{rank=} sending to left {left=}")
    #         sendbuf[:] = x[num_ghost : 2 * num_ghost]
    #         comm.Sendrecv(
    #             sendbuf,
    #             dest=left,
    #             sendtag=TAG,
    #             recvbuf=recvbuf,
    #             source=left,
    #             recvtag=TAG,
    #         )
    #         x[num_ghost : 2 * num_ghost, :] += recvbuf
    #         # print(f"{rank=} received from left")

    #     if right != MPI.PROC_NULL:
    #         # print(f"{rank=} sending to right {right=}")
    #         sendbuf[:] = x[-2 * num_ghost : -num_ghost]
    #         comm.Sendrecv(
    #             sendbuf,
    #             dest=right,
    #             sendtag=TAG,
    #             recvbuf=recvbuf,
    #             source=right,
    #             recvtag=TAG,
    #         )
    #         x[-2 * num_ghost : -num_ghost, :] += recvbuf
    #         # print(f"{rank=} received from right")

    # else:
    #     # odd ranks do right → left
    #     if right != MPI.PROC_NULL:
    #         # print(f"{rank=} sending to right {right=}")
    #         sendbuf[:] = x[-2 * num_ghost : -num_ghost, :]
    #         comm.Sendrecv(
    #             sendbuf,
    #             dest=right,
    #             sendtag=TAG,
    #             recvbuf=recvbuf,
    #             source=right,
    #             recvtag=TAG,
    #         )
    #         x[-2 * num_ghost : -num_ghost] += recvbuf
    #         # print(f"{rank=} received from right")

    #     if left != MPI.PROC_NULL:
    #         # print(f"{rank=} sending to left {left=}")
    #         sendbuf[:] = x[num_ghost : 2 * num_ghost, :]
    #         comm.Sendrecv(
    #             sendbuf,
    #             dest=left,
    #             sendtag=TAG,
    #             recvbuf=recvbuf,
    #             source=left,
    #             recvtag=TAG,
    #         )
    #         x[num_ghost : 2 * num_ghost, :] += recvbuf
    #         # print(f"{rank=} received from left")

    # for r in range(size):
    #     comm.Barrier()
    #     if rank == r:
    #         print(f"Rank {rank}: {x[num_ghost : -num_ghost, :] = }")


# Rank-count agnostic: these run serially and under any ``mpirun -n N``.


def _as_numpy(data) -> np.ndarray:
    """Return backend data as a NumPy array for assertions."""
    return data.get() if hasattr(data, "get") else np.asarray(data)


@pytest.mark.parametrize("filter_name", ["binomial", "gaussian"])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("passes", [1, 3])
@pytest.mark.parametrize("component", [None, 1])
def test_halo_smoothing_matches_gathered_smoothing(
    filter_name: str,
    periodic: bool,
    passes: int,
    component: int | None,
) -> None:
    """Smoothing a decomposed 2D field equals smoothing the whole array serially."""
    rng = np.random.default_rng(11)
    global_data = rng.normal(size=(16, 12, 2))
    kwargs = {
        "lower_bound": (0.0, 0.0),
        "upper_bound": (1.0, 1.0),
        "periodic": (periodic, periodic),
    }
    reference = Field.from_array(xp.asarray(global_data), comm=None, **kwargs)
    reference.smooth(filter_name=filter_name, passes=passes, component=component)

    decomposed = Field.from_array(
        xp.asarray(global_data),
        comm=MPI.COMM_WORLD,
        domain_decomposition=[True, True],
        num_ghost_points=1,
        **kwargs,
    )
    radius = 2 if filter_name == "gaussian" else 1
    if decomposed.field_data.layout.distributed:
        assert decomposed._can_smooth_with_halos(radius)
    decomposed.smooth(filter_name=filter_name, passes=passes, component=component)

    np.testing.assert_allclose(
        decomposed.to_numpy(),
        _as_numpy(reference.to_ndarray()),
        rtol=1e-12,
        atol=1e-14,
    )
    halos = np.ones(decomposed.field_data.layout.storage_shape, dtype=bool)
    halos[decomposed.field_data.layout.interior] = False
    assert not _as_numpy(decomposed.local_with_halos)[halos].any()


def test_smoothing_falls_back_when_blocks_are_narrower_than_the_kernel() -> None:
    """With fewer cells per rank than the kernel radius, the gathered path is used."""
    size = MPI.COMM_WORLD.Get_size()
    # One cell per rank is narrower than the Gaussian kernel's radius of 2.
    global_data = np.arange(float(max(size, 2))) ** 2
    kwargs = {"lower_bound": (0.0,), "upper_bound": (1.0,), "periodic": True}
    reference = Field.from_array(xp.asarray(global_data), comm=None, **kwargs)
    reference.smooth_gaussian(passes=2)

    decomposed = Field.from_array(
        xp.asarray(global_data),
        comm=MPI.COMM_WORLD,
        domain_decomposition=[True],
        **kwargs,
    )
    if size > 1:
        assert not decomposed._can_smooth_with_halos(2)
    decomposed.smooth_gaussian(passes=2)

    np.testing.assert_allclose(
        decomposed.to_numpy(),
        _as_numpy(reference.to_ndarray()),
    )


def test_set_field_data_component_writes_locally_and_zeroes_halos() -> None:
    """Setting one component from a global array needs no gather."""
    global_data = np.arange(10 * 8 * 3, dtype=float).reshape(10, 8, 3)
    field = Field.from_array(
        xp.asarray(global_data),
        lower_bound=(0.0, 0.0),
        upper_bound=(1.0, 1.0),
        comm=MPI.COMM_WORLD,
        domain_decomposition=[True, True],
        num_ghost_points=1,
    )
    field.local_with_halos[...] += 1000.0
    replacement = -np.arange(80, dtype=float).reshape(10, 8)

    field.set_field_data(xp.asarray(replacement), component=1)

    expected = global_data.copy()
    expected[..., 0] += 1000.0
    expected[..., 2] += 1000.0
    expected[..., 1] = replacement
    np.testing.assert_array_equal(field.to_numpy(), expected)
    halos = np.ones(field.field_data.layout.storage_shape, dtype=bool)
    halos[field.field_data.layout.interior] = False
    assert not _as_numpy(field.local_with_halos)[halos].any()

    scalar = Field.from_array(
        xp.zeros((10, 8)),
        lower_bound=(0.0, 0.0),
        upper_bound=(1.0, 1.0),
        comm=MPI.COMM_WORLD,
        domain_decomposition=[True, True],
    )
    scalar.set_field_data(xp.asarray(replacement))
    np.testing.assert_array_equal(scalar.to_numpy()[..., 0], replacement)


@pytest.mark.parametrize("periodic", [(False, False), (True, False)])
def test_gradient_does_not_depend_on_the_split(periodic: tuple[bool, bool]) -> None:
    rng = np.random.default_rng(3)
    global_data = rng.normal(size=(12, 10))
    kwargs = {
        "lower_bound": (0.0, 0.0),
        "upper_bound": (1.0, 2.0),
        "periodic": periodic,
    }
    reference = Field.from_array(global_data, comm=None, **kwargs).gradient()

    decomposed = Field.from_array(
        global_data,
        comm=MPI.COMM_WORLD,
        domain_decomposition=[True, True],
        num_ghost_points=1,
        **kwargs,
    ).gradient()

    assert decomposed.num_components == 2
    np.testing.assert_allclose(decomposed.to_numpy(), reference.to_numpy(), rtol=1e-12)


def test_set_inside_marks_the_same_nodes_on_any_split() -> None:
    from simgeometry.objects import Circle

    kwargs = {"ndim": 2, "lower_bound": (-1.0, -1.0), "upper_bound": (1.0, 1.0)}
    circle = Circle(midpoint=(0.2, -0.1), radius=0.55)
    reference = Field(num_gridpoints=(17, 13), **kwargs)
    reference.set_inside(circle, 1.0)

    decomposed = Field(
        num_gridpoints=(17, 13),
        comm=MPI.COMM_WORLD,
        domain_decomposition=[True, True],
        num_ghost_points=1,
        **kwargs,
    )
    decomposed.set_inside(circle, 1.0)

    np.testing.assert_array_equal(decomposed.to_numpy(), reference.to_numpy())
    assert reference.sum() > 0


def test_deposit_then_exchange_halos_equals_the_serial_deposit() -> None:
    """Each rank deposits its own markers; the halo exchange completes the sum."""
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(4)
    positions = rng.uniform(0.0, 1.0, (2, 2000))
    values = rng.uniform(0.5, 1.5, 2000)
    kwargs = {"ndim": 2, "lower_bound": (0.0, 0.0), "upper_bound": (1.0, 1.0)}
    serial = Field(num_gridpoints=(21, 17), **kwargs)
    serial.scatter(positions, values)

    field = Field(
        num_gridpoints=(21, 17),
        comm=comm,
        domain_decomposition=[True, False],
        num_ghost_points=1,
        **kwargs,
    )
    # the markers of this rank: those in its block of nodes (the last cell
    # of a block reaches into the halo)
    grid = field.block_grid(cpu=True)
    lower = grid.bounds.lower_bound[0] + grid.grid_spacing[0]  # first own node
    upper = lower + (field.local.shape[0]) * grid.grid_spacing[0]
    is_last = field.proc_index_bounds[0][1] == field.num_gridpoints[0]
    mine = (positions[0] >= lower) & (
        (positions[0] <= upper) if is_last else (positions[0] < upper)
    )
    field.scatter(np.ascontiguousarray(positions[:, mine]), values[None, mine])
    field.exchange_halos()

    np.testing.assert_allclose(field.to_numpy(), serial.to_numpy(), rtol=1e-12)
    assert comm.allreduce(int(mine.sum())) == 2000


def _own_markers(field: Field, positions: np.ndarray) -> np.ndarray:
    """Which markers lie in this rank's cells: from its first node to the next rank's.

    The first and last rank of a bounded axis also take the markers beyond the
    domain edge, which deposit on the edge nodes as on an unsplit field.
    """
    mine = np.ones(positions.shape[1], dtype=bool)
    spacing = np.asarray(xp.to_numpy(field.grid_spacing))
    lower_bound = np.asarray(xp.to_numpy(field.lower_bound))
    periodic = field.periodic if field.periodic is not None else (False,) * field.ndim
    for axis in range(field.ndim):
        start, stop = field.proc_index_bounds[axis]
        x = positions[axis]
        lower = lower_bound[axis] + start * spacing[axis]
        upper = lower_bound[axis] + stop * spacing[axis]
        bounded = not periodic[axis]
        above = (x >= lower) | (bounded and start == 0)
        last = stop == field.num_gridpoints[axis]
        below = (x < upper) | (bounded and last)
        mine &= above & below
    return mine


@pytest.mark.parametrize(
    ("decomposition", "periodic"),
    [
        ([True, False], (False, False)),
        ([True, False], (True, False)),
        ([True, True], (False, True)),
    ],
)
@pytest.mark.parametrize("weighting", [LINEAR, QUADRATIC], ids=lambda w: w.name)
def test_split_deposit_and_gather_equal_the_serial_ones(
    weighting, decomposition, periodic
) -> None:
    """Each rank deposits and gathers at its own markers, also near the domain edges.

    After the halo exchange the deposit is the serial one: no charge is lost in
    the halo nodes beyond a bounded edge, where a quadratic stencil reaches.
    After filling the halos each rank's gather is the serial one.
    """
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(14)
    positions = rng.uniform(-0.05, 1.05, (2, 3000))
    positions[:, :40] = rng.uniform(0.0, 0.02, (2, 40))  # by the lower edges
    for axis, flag in enumerate(periodic):
        if flag:
            positions[axis] %= 1.0
    values = rng.uniform(0.5, 1.5, (2, 3000))
    kwargs = {
        "ndim": 2,
        "lower_bound": (0.0, 0.0),
        "upper_bound": (1.0, 1.0),
        "num_gridpoints": (24, 20),
        "num_components": 2,
        "periodic": periodic,
    }
    serial = Field(**kwargs)
    serial.scatter(positions, values, weighting)

    field = Field(
        comm=comm,
        domain_decomposition=decomposition,
        num_ghost_points=weighting.halo,
        **kwargs,
    )
    mine = _own_markers(field, positions)
    assert comm.allreduce(int(mine.sum())) == positions.shape[1]
    own = np.ascontiguousarray(positions[:, mine])
    field.scatter(own, np.ascontiguousarray(values[:, mine]), weighting)
    field.exchange_halos()
    np.testing.assert_allclose(field.to_numpy(), serial.to_numpy(), rtol=1e-12)

    potential = np.sin(3 * np.arange(24))[:, None, None] + np.cos(np.arange(20))[
        None, :, None
    ] * np.array([1.0, -2.0])
    serial.set_field_data(potential)
    field.set_field_data(potential)
    field.fill_halos()
    np.testing.assert_allclose(
        np.asarray(xp.to_numpy(field.gather(own, weighting))),
        np.asarray(xp.to_numpy(serial.gather(own, weighting))),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.mpi(min_size=2)
def test_a_weighting_wider_than_the_halo_raises() -> None:
    field = Field(
        ndim=2,
        lower_bound=(0.0, 0.0),
        upper_bound=(1.0, 1.0),
        num_gridpoints=(24, 20),
        comm=MPI.COMM_WORLD,
        domain_decomposition=[True, False],
        num_ghost_points=1,
    )
    positions = np.full((2, 1), 0.5)
    field.scatter(positions, 1.0)  # linear needs one halo node
    with pytest.raises(ValueError, match="num_ghost_points >= 2"):
        field.scatter(positions, 1.0, QUADRATIC)
    with pytest.raises(ValueError, match="num_ghost_points >= 2"):
        field.gather(positions, QUADRATIC)


def test_export_and_load_under_mpi(tmp_path_factory: pytest.TempPathFactory) -> None:
    comm = MPI.COMM_WORLD
    # one file for every rank: rank 0 picks the directory
    directory = comm.bcast(
        str(tmp_path_factory.mktemp("export")) if comm.Get_rank() == 0 else None
    )
    global_data = np.arange(9 * 6, dtype=float).reshape(9, 6)
    field = Field.from_array(
        global_data,
        lower_bound=(0.0, 0.0),
        upper_bound=(1.0, 1.0),
        comm=comm,
        domain_decomposition=[True, True],
    )
    field.export(f"{directory}/field.nc")

    loaded = Field.load(
        f"{directory}/field.nc", comm=comm, domain_decomposition=[True, False]
    )

    np.testing.assert_array_equal(loaded.to_numpy()[..., 0], global_data)
