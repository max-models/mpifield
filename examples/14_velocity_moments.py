"""Velocity moments

Deposit the density, flow, temperature and heat flux of markers that sample a non-uniform drifting Maxwellian.

`Field.deposit_density` deposits marker values and divides them by the volume each node
collects (`Field.node_volumes`: a cell, half of it on a bounded edge), so the result is a
density in physical units. With the products $w$, $w v$, $w v^2$, $w v^3$ as four
components it gives the raw velocity moments $n$, $n u$, $n\\langle v^2\\rangle$,
$n\\langle v^3\\rangle$ in one deposit (mpiparticles' `moment_values` builds these products
for any order and number of velocity components). The central moments follow: the
temperature over the mass $T = \\langle v^2\\rangle - u^2$ and the heat flux
$q = n\\langle (v - u)^3\\rangle / 2 = (M_3 - 3 u M_2 + 2 n u^3) / 2$. The markers sample
$n(x) = 1 + x/2$, $u(x) = \\sin \\pi x$ and $T(x) = 1 + x^2$ on a bounded grid, so the heat
flux is zero: what remains is the noise of $2 \\times 10^5$ markers.
"""

import matplotlib.pyplot as plt
import numpy as np

from mpifield import Field

rng = np.random.default_rng(2)
count = 200_000
moments = Field(
    ndim=1, lower_bound=(0.0,), upper_bound=(1.0,), num_gridpoints=41, num_components=4
)

# positions from n(x) = 1 + x / 2 (inverse transform), velocities from the local Maxwellian
uniform = rng.uniform(0.0, 1.0, count)
x = 2.0 * (np.sqrt(1.0 + 1.25 * uniform) - 1.0)
v = np.sin(np.pi * x) + np.sqrt(1.0 + x**2) * rng.normal(size=count)
weights = np.full(count, 1.25 / count)  # total: the integral of n

moments.deposit_density(x[None, :], np.stack([weights * v**k for k in range(4)]))
n, m1, m2, m3 = moments.to_numpy().T
u = m1 / n
temperature = m2 / n - u**2
heat_flux = (m3 - 3 * u * m2 + 2 * n * u**3) / 2

nodes = np.linspace(0.0, 1.0, 41)
exact = {
    "density $n$": (n, 1 + nodes / 2),
    "flow $u$": (u, np.sin(np.pi * nodes)),
    "temperature $T/m$": (temperature, 1 + nodes**2),
    "heat flux $q$": (heat_flux, 0 * nodes),
}
fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, layout="constrained")
for ax, (label, (deposited, expected)) in zip(axes.flat, exact.items(), strict=True):
    ax.plot(nodes, deposited, "o", ms=3, label="deposited")
    ax.plot(nodes, expected, "k--", label="exact")
    ax.set_title(label)
for ax in axes[1]:
    ax.set_xlabel("$x$")
axes[0, 0].legend(frameon=False)
print(f"max |T - exact| = {np.abs(temperature - (1 + nodes**2)).max():.3f}")
plt.show()
