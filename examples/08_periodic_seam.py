"""A periodic axis

On a periodic axis the last node neighbours the first: deposits and gathers wrap across the seam.

A periodic axis with $n$ nodes has $n$ cells over its length, so the node spacing is
$L/n$ and no node sits on the upper bound. A marker in the last cell deposits onto the last
and the first node, and a gather there blends the same two nodes. Here a ring of markers
crossing the seam in $y$ deposits a density that is continuous across it, and
`Field.evaluate` repeats the first node at the upper bound to interpolate across the seam.
"""

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots  # noqa: F401

from mpifield import Field

rng = np.random.default_rng(8)
density = Field(
    ndim=2,
    lower_bound=(0.0, 0.0),
    upper_bound=(1.0, 1.0),
    num_gridpoints=(41, 40),
    periodic=(False, True),
)
print(f"spacing along y: {float(density.grid_spacing[1]):.4f} (1/40, not 1/39)")
count = 30_000
angle = rng.uniform(0, 2 * np.pi, count)
radius = 0.3 + 0.03 * rng.normal(size=count)
# a ring centred on the seam y = 0 = 1, wrapped into [0, 1)
positions = np.stack(
    [0.5 + radius * np.cos(angle), np.mod(radius * np.sin(angle), 1.0)]
)
density.scatter_linear(positions, np.ones((1, count)))

density.plot(name="markers per node", equal_aspect=True, title="a ring across the seam")
density.plot(x=0.5, y=np.linspace(0.0, 1.0, 400), name="markers per node")
plt.show()
