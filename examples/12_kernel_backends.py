"""Kernel backends

The same deposit and gather through the compiled Pyccel kernels and their NumPy versions, timed.

Every kernel has a Pyccel version (compiled with OpenMP on first use), a NumPy version (used
without Pyccel), a CUDA version (for CuPy arrays) and a Metal version (Apple-silicon GPUs,
float32). `cunumpy.kernels.use_host_kernel_implementation` picks the host version; the
results agree to round-off. The plot shows the time per marker of a 2D deposit and gather
for both host versions as the marker count grows, the best of five runs, measured where
the docs were built.
"""

import time
from functools import partial

import cunumpy as xp
import matplotlib.pyplot as plt
import numpy as np
import plasma_plots  # noqa: F401
import xarray as xr

from mpifield import Field

rng = np.random.default_rng(12)


def best(call, repeats=5):
    """The shortest of a few timings, the least disturbed by the machine."""
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        durations.append(time.perf_counter() - start)
    return min(durations)


kwargs = {"ndim": 2, "lower_bound": (0.0, 0.0), "upper_bound": (1.0, 1.0)}
counts = np.array([10_000, 100_000, 1_000_000])
times = np.zeros((2, 2, counts.size))  # implementation, operation, count
for c, count in enumerate(counts):
    positions = rng.random((2, count))
    values = np.ones((1, count))
    for i, implementation in enumerate(("pyccel", "numpy")):
        with xp.kernels.use_host_kernel_implementation(implementation):
            field = Field(num_gridpoints=(257, 257), **kwargs)
            field.scatter_linear(positions, values)  # compile and warm up
            times[i, 0, c] = (
                best(partial(field.scatter_linear, positions, values)) / count
            )
            times[i, 1, c] = best(partial(field.interpolate_field, positions)) / count

per_marker = xr.DataArray(
    times * 1e9,
    dims=("implementation", "operation", "markers"),
    coords={
        "implementation": ["pyccel", "numpy"],
        "operation": ["deposit", "gather"],
        "markers": counts,
    },
    name="time per marker",
    attrs={"units": "ns"},
)
print(per_marker.sel(markers=counts[-1]).to_pandas())
fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
for implementation in ("pyccel", "numpy"):
    for operation in ("deposit", "gather"):
        selected = per_marker.sel(implementation=implementation, operation=operation)
        selected.plasma.plot.lineout(x="markers", ax=ax)
        ax.lines[-1].set_label(f"{implementation} {operation}")
ax.set(xscale="log", yscale="log", title="2D, 257 x 257 nodes")
ax.legend(frameon=False)
plt.show()
