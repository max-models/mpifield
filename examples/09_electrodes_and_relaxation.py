"""Electrodes from shapes

Mark electrodes with simgeometry shapes and relax Laplace's equation with field arithmetic.

`Field.set_inside` sets the nodes inside a shape, here two plates and a rod at fixed
voltages. Jacobi iterations of Laplace's equation, written with the stored arrays of the
fields, then fill in the potential between them: each sweep replaces a node by the mean of
its four neighbours and restores the electrodes. This is not a fast solver (patchwork and
picnic have those) but shows the field as a plain array container.
"""

import matplotlib.pyplot as plt
import plasma_plots  # noqa: F401
from simgeometry.objects import Circle, Rectangle

from mpifield import Field

kwargs = {"ndim": 2, "lower_bound": (0.0, 0.0), "upper_bound": (0.1, 0.06)}
shape = (101, 61)
electrodes = {
    "lower plate": (Rectangle(midpoint=(0.05, 0.002), side_lengths=(0.1, 0.004)), 0.0),
    "upper plate": (Rectangle(midpoint=(0.05, 0.058), side_lengths=(0.1, 0.004)), 0.0),
    "rod": (Circle(midpoint=(0.05, 0.03), radius=0.008), 1000.0),
}
fixed = Field(num_gridpoints=shape, **kwargs)
voltage = Field(num_gridpoints=shape, **kwargs)
for shape_, value in electrodes.values():
    fixed.set_inside(shape_, 1.0)
    voltage.set_inside(shape_, value)
mask = fixed.local[..., 0] > 0

phi = voltage.copy()
data = phi.local[..., 0]
for _ in range(4000):
    interior = 0.25 * (
        data[:-2, 1:-1] + data[2:, 1:-1] + data[1:-1, :-2] + data[1:-1, 2:]
    )
    data[1:-1, 1:-1] = interior
    data[mask] = voltage.local[..., 0][mask]

phi.plot(name="potential", units="V", levels=24, equal_aspect=True)
field = -phi.gradient()
field.magnitude().plot(name="|E|", units="V/m", equal_aspect=True, robust=True)
plt.show()
