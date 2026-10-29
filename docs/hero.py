"""The figure on the docs home page: field lines of a dipole on a grid split over four ranks.

A potential with a positive and a negative charge is set on a :class:`mpifield.Field` that is
split over the MPI ranks, its gradient gives the electric field, and markers started around the
positive charge follow the field: every step each rank gathers the field at its own markers
(``Field.gather``) and the markers that crossed into another rank's block move there
(``mpiarray.migrate``). The rank blocks, the field lines the markers traced and dots moving along
them are drawn with ``plasma_plots.hero`` into ``docs/src/assets/hero.svg``. Rerun with::

    mpiexec -n 4 python docs/hero.py   # or: make docs-hero
"""

from pathlib import Path

import maybempi
import mpiarray as mpa
import numpy as np
from plasma_plots.hero import HeroFigure

from mpifield import Field

comm = maybempi.get_mpi().COMM_WORLD
rank = comm.Get_rank()
lower, upper, nodes = (0.0, 0.0), (8.0, 3.0), (161, 61)
charges = [(4.9, 1.5, 1.0), (6.9, 1.6, -1.0)]  # x, y, charge

phi = Field(
    ndim=2,
    lower_bound=lower,
    upper_bound=upper,
    num_gridpoints=nodes,
    comm=comm,
    num_ghost_points=1,
)
x, y = np.meshgrid(*phi.get_grid_points(), indexing="ij")
phi.set_field_data(
    sum(-q * np.log((x - cx) ** 2 + (y - cy) ** 2 + 0.01) for cx, cy, q in charges)
)
E = -phi.gradient()
E.fill_halos()
layout = E.field_data.layout
spacing = (np.asarray(upper) - np.asarray(lower)) / (np.asarray(nodes) - 1)

# markers on a small circle around the positive charge, started on rank 0 and sent to their owners
count, step, steps = 26, 0.04, 220
angle = 2 * np.pi * (np.arange(count) + 0.5) / count
start = np.stack([4.9 + 0.12 * np.cos(angle), 1.5 + 0.12 * np.sin(angle)], axis=1)
positions = start if rank == 0 else np.zeros((0, 2))
ids = np.arange(count) if rank == 0 else np.zeros(0, dtype=np.int64)


def owners(points):
    """Return the rank that holds the node left of and below each point."""
    cells = np.clip(
        ((points - lower) / spacing).astype(np.int64), 0, np.asarray(nodes) - 1
    )
    return layout.owners(
        np.concatenate([cells, np.zeros((len(points), 1), np.int64)], axis=1)
    )


history = []
for _ in range(steps):
    positions, ids = mpa.migrate(owners(positions), positions, ids, comm=comm)
    positions, ids = np.asarray(positions), np.asarray(ids)
    history.append((ids.copy(), positions.copy(), np.full(len(ids), rank)))
    field = np.asarray(E.gather(positions.T)).T if len(ids) else np.zeros((0, 2))
    norm = np.linalg.norm(field, axis=1, keepdims=True)
    near_sink = np.hypot(positions[:, 0] - 6.9, positions[:, 1] - 1.6) < 0.08
    moved = positions + step * field / np.where(norm > 0, norm, 1)
    inside = np.all((moved >= lower) & (moved < upper), axis=1)  # stop at the walls
    moving = (~near_sink & inside & (norm[:, 0] > 0))[:, None]
    positions = np.where(moving, moved, positions)

history = comm.gather(history, root=0)
grid = phi.block_grid(cpu=True, inside_domain=True)
blocks = comm.gather([np.asarray(b).tolist() for b in grid.bounds], root=0)
if rank == 0:
    tracks = {i: [] for i in range(count)}
    for rank_history in history:
        for t, (ids_t, pos_t, _) in enumerate(rank_history):
            for i, p in zip(ids_t, pos_t, strict=True):
                tracks[int(i)].append((t, *p))
    fig = HeroFigure(xlim=(0, 8), ylim=(0, 3))
    for r, ((x0, y0), (x1, y1)) in enumerate(blocks):
        # neighbouring blocks share a cell: draw the rank border in its middle
        x0, y0 = (
            v + h / 2 if v > lo else v
            for v, h, lo in zip((x0, y0), spacing, lower, strict=True)
        )
        x1, y1 = (
            v - h / 2 if v < hi else v
            for v, h, hi in zip((x1, y1), spacing, upper, strict=True)
        )
        fig.rect(
            x0,
            y0,
            x1,
            y1,
            tone="low",
            opacity=0.35 if r % 3 else 0.7,
            outline="muted",
            title=f"rank {r}",
        )
    for i, track in tracks.items():
        track = np.array(sorted(track))[:, 1:]
        moved = np.r_[True, np.any(np.diff(track, axis=0) != 0, axis=1)]
        track = track[moved][::3]
        line = fig.line(
            track[:, 0], track[:, 1], tone="accent", width=1.2, opacity=0.45
        )
        length = len(track) * 3 * step
        dot = fig.tracer(line, duration=1.6 * length + 2, offset=(i * 0.77) % 6, r=3.5)
        dot.animate(
            "opacity", [0, 1, 1, 1, 1, 0], 1.6 * length + 2, begin=(i * 0.77) % 6
        )
    for cx, cy, q in charges:
        fig.disc(
            cx,
            cy,
            0.05,
            outline="high" if q > 0 else "muted",
            title="positive charge" if q > 0 else "negative charge",
        )
    fig.save(Path(__file__).parent / "src" / "assets" / "hero.svg")
