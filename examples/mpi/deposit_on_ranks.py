"""Deposit and gather on several MPI ranks.

Run with ``mpiexec -n 4 python examples/mpi/deposit_on_ranks.py``. Each rank owns a block of
the grid (plus one halo node on every side) and the markers inside it. It deposits its
markers into its block, ``exchange_halos`` adds the halo contributions to the neighbours'
nodes, and the gathered result equals the serial deposit. For the gather the halos are
filled from the neighbours first, so markers in a block's last cell see the next node.
"""

import numpy as np
from maybempi import MPI

from mpifield import Field

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
kwargs = {"ndim": 2, "lower_bound": (0.0, 0.0), "upper_bound": (1.0, 1.0)}

# the same markers on every rank (the seed), each rank keeps those in its block
rng = np.random.default_rng(0)
positions = rng.random((2, 100_000))

density = Field(
    num_gridpoints=(129, 129),
    comm=comm,
    domain_decomposition=[True, True],
    num_ghost_points=1,
    **kwargs,
)
grid = density.block_grid(cpu=True)
first = grid.bounds.lower_bound + grid.grid_spacing  # the first own node, per axis
end = first + np.array(density.local.shape[:2]) * grid.grid_spacing
last_block = [
    density.proc_index_bounds[d][1] == density.num_gridpoints[d] for d in range(2)
]
mine = np.ones(positions.shape[1], dtype=bool)
for d in range(2):
    upper = positions[d] <= end[d] if last_block[d] else positions[d] < end[d]
    mine &= (positions[d] >= first[d]) & upper
local = np.ascontiguousarray(positions[:, mine])

density.scatter(local, np.ones((1, local.shape[1])))
density.exchange_halos()
total = density.sum()

density.fill_halos()
gathered = density.gather(local)

serial = Field(num_gridpoints=(129, 129), **kwargs)
serial.scatter(positions, np.ones((1, positions.shape[1])))
error = np.abs(density.to_numpy() - serial.to_numpy()).max()
if rank == 0:
    print(f"{size} ranks: total {total:.0f}, largest difference to serial {error:.1e}")
print(
    f"rank {rank}: {local.shape[1]} markers, block {density.local.shape[:2]}, "
    f"mean gathered density {gathered.mean():.2f}"
)
