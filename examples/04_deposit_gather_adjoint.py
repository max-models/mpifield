"""Adjoint deposit and gather

Deposit and gather are transposes: the scatter of marker values and the gather of a field use the same weights, so they are adjoint.

For any field $g$ on the nodes and charges $q_i$ at positions $x_i$,
$\\sum_i q_i\\, g(x_i) = \\sum_n g_n \\rho_n$ with $\\rho$ the deposit of the charges. That is
what makes a PIC code with linear weights momentum-conserving and free of self-forces on a
uniform grid. The script checks the identity on random data, then moves one marker across a
cell and gathers the field of its own (smoothed) deposit at it: the self-force vanishes to
round-off, about $10^{-17}$, wherever the marker is.
"""

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots  # noqa: F401
import xarray as xr

from mpifield import Field

rng = np.random.default_rng(4)
shape = (33, 25)
kwargs = {"ndim": 2, "lower_bound": (0.0, 0.0), "upper_bound": (1.0, 0.75)}
g = Field(num_gridpoints=shape, **kwargs)
g.set_field_data(rng.normal(size=shape))
rho = Field(num_gridpoints=shape, **kwargs)
positions = rng.random((2, 1000)) * np.array([[1.0], [0.75]])
charges = rng.normal(size=1000)
rho.scatter(positions, charges)

left = float(np.dot(g.gather(positions)[0], charges))
right = float((g * rho).sum())
print(f"sum_i q_i g(x_i) = {left:.12f}")
print(f"sum_n g_n rho_n  = {right:.12f}  (difference {abs(left - right):.1e})")

# the gather of a marker's own 'potential': the deposit smoothed by a stencil
offsets = np.linspace(0.0, 1.0, 101)
dx = float(g.grid_spacing[0])
force = []
for offset in offsets:
    one = Field(num_gridpoints=shape, **kwargs)
    x = np.array([[0.5 + offset * dx], [0.375]])
    one.scatter(x, 1.0)
    potential = one.smooth_binomial(passes=1, in_place=False)
    field = -potential.gradient()
    force.append(float(field.gather(x)[0, 0]))
self_force = xr.DataArray(
    force, dims="offset", coords={"offset": offsets}, name="self-force"
)
result = self_force.plasma.plot.lineout(x="offset")
result.ax.set(xlabel="position in the cell (cells)", title="a marker's force on itself")
plt.show()
