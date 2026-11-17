"""The live demo of the mpifield docs: the density of markers, deposited onto a 1D grid.

Markers sampled from n(x) = 1 + 0.5 cos(x) on a periodic domain of length 2 pi are
deposited with `Field.deposit_density`, linear (cloud-in-cell) or quadratic, and
optionally smoothed with the binomial filter. It runs in the browser (Pyodide), on the
NumPy versions of the kernels; `run` takes the values of the controls.
"""

import numpy as np

from mpifield import LINEAR, QUADRATIC, Field

LENGTH = 2 * np.pi
AMPLITUDE = 0.5
WEIGHTINGS = {"linear": LINEAR, "quadratic": QUADRATIC}


def _density(x):
    return 1.0 + AMPLITUDE * np.cos(x)


def _sample(num_markers, rng):
    """Draw marker positions from the density by inverting its cumulative sum."""
    grid = np.linspace(0.0, LENGTH, 2001)
    cumulative = grid + AMPLITUDE * np.sin(grid)
    return np.interp(rng.random(num_markers) * LENGTH, cumulative, grid)


def run(markers=10_000, nodes=32, weighting="linear", passes=0):
    """Deposit the markers and compare the density with the exact one.

    Args:
        markers: Number of markers; each carries the weight ``2 pi / markers``, so
            that the mean density is 1.
        nodes: Number of grid nodes (cells) on the periodic domain.
        weighting: The shape function of the deposit, ``"linear"`` or
            ``"quadratic"``.
        passes: Passes of the binomial filter ``[1, 2, 1] / 4`` after the deposit.

    Returns:
        The figure and the numbers, as the live demo takes them.
    """
    markers, nodes, passes = int(markers), int(nodes), int(passes)
    positions = _sample(markers, np.random.default_rng(1))
    density = Field(
        ndim=1,
        lower_bound=(0.0,),
        upper_bound=(LENGTH,),
        num_gridpoints=nodes,
        periodic=True,
    )
    density.deposit_density(
        positions[np.newaxis], np.full(markers, LENGTH / markers), WEIGHTINGS[weighting]
    )
    if passes > 0:
        density.smooth_binomial(passes=passes)
    (x,) = density.get_grid_points()
    values = density.to_numpy()[:, 0]
    total = float(values.sum() * density.grid_spacing[0])
    error = float(np.sqrt(np.mean((values - _density(x)) ** 2)))
    fine = np.linspace(0.0, LENGTH, 400)
    return {
        "figure": {
            "data": [
                {"x": fine, "y": _density(fine), "mode": "lines", "name": "exact"},
                {
                    "x": x,
                    "y": values,
                    "mode": "lines+markers",
                    "name": f"{weighting} deposit",
                },
            ],
            "layout": {
                "xaxis": {"title": {"text": "x"}, "range": [0.0, LENGTH]},
                "yaxis": {"title": {"text": "density"}, "range": [0.0, 2.0]},
            },
        },
        "numbers": {
            "Total weight (exactly 2π)": total,
            "RMS error": error,
            "Expected noise": float(np.sqrt(nodes / markers)),
        },
    }
