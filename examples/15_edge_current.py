"""Edge fields and currents

Deposit the current of moving markers on the edges of a grid with `WHITNEY`: the change of their linear charge, exactly.

An edge field holds the line integral of a vector field along every edge of the grid,
``num_nodes + (ndim,)``; inside a cell the field is the lowest-order edge (Whitney)
interpolant. `WHITNEY.deposit_paths_block` cuts each marker's path where it crosses a grid
line and integrates the edge functions along every piece, so the discrete divergence of the
current, `WHITNEY.divergence_block`, is the change of `LINEAR`'s charge between the starts
and the ends of the paths, to round-off. The current times an edge field is the work of
its field on the markers: for the gradient of a potential $\\phi$, the sum of
$q\\,(\\phi(x_1) - \\phi(x_0))$. These are the identities an energy- and charge-conserving
implicit PIC is built on. The plot shows the change of the charge along one row of the
grid, deposited at the two ends, and the divergence of the current.
"""

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots  # noqa: F401  (registers the .plasma accessor)
import xarray as xr

from mpifield import LINEAR, WHITNEY, BlockGrid

rng = np.random.default_rng(15)
shape = (32, 24)
grid = BlockGrid.from_geometry((0.0, 0.0), (1.0 / 32, 1.0 / 24), shape, (True, True))
count = 4000
start = rng.random((2, count))
velocity = rng.normal(0.0, 0.05, (2, count))
end = start + velocity  # not wrapped: the deposit follows the paths across the seam
charges = rng.choice([-1.0, 1.0], count)

current = np.zeros((*shape, 2))
WHITNEY.deposit_paths_block(start, end, charges, current, grid)


def charge_at(points):
    """`LINEAR`'s deposit of the markers' charges at ``points`` on the nodes."""
    charge = np.zeros((*shape, 1))
    LINEAR.scatter_block(points, charges, charge, grid)
    return charge[..., 0]


change = charge_at(end) - charge_at(start)
divergence = WHITNEY.divergence_block(current, grid)
print(f"largest |G^T j - (Q(x1) - Q(x0))| = {np.abs(divergence - change).max():.1e}")

potential = rng.normal(size=shape)
field = WHITNEY.gradient_block(potential, grid)
phi_start, phi_end = np.zeros((1, count)), np.zeros((1, count))
LINEAR.gather_block(start, potential[..., None], phi_start, grid)
LINEAR.gather_block(end, potential[..., None], phi_end, grid)
work = float(np.sum(current * field))
expected = float(np.sum(charges * (phi_end - phi_start)))
print(f"j . G phi = {work:.12f}, sum q (phi(x1) - phi(x0)) = {expected:.12f}")

x = np.arange(shape[0]) / shape[0]
row = shape[1] // 2
fig, ax = plt.subplots()
for values, name in [
    (change[:, row], r"$Q(x_1) - Q(x_0)$"),
    (divergence[:, row], r"$G^\top j$"),
]:
    xr.DataArray(
        values, dims="x", coords={"x": x}, name="charge", attrs={"long_name": name}
    ).plasma.plot.lineout(x="x", ax=ax)
    ax.lines[-1].set_label(name)
ax.lines[-1].set_linestyle("--")
ax.set(xlabel="$x$", ylabel="change of the node charge", title="one row of the grid")
ax.legend(frameon=False)
plt.show()
