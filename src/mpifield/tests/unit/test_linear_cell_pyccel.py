"""The linear-cell blocks of the Pyccel kernels against the NumPy version."""

import numpy as np
import pytest

from mpifield.kernels.linear_cell import linear_cell
from mpifield.kernels.linear_cell_pyccel import (
    cell_fraction,
    left_node,
    periodic_position,
    right_node,
)


@pytest.mark.parametrize("periodic", [0, 1])
def test_the_pyccel_blocks_are_the_numpy_cell(periodic):
    # run in Python: positions inside, on the nodes and beyond both ends
    num = 9
    rel = np.concatenate(
        [
            np.random.default_rng(1).uniform(-3.0, num + 3.0, 200),
            np.arange(-1.0, num + 2),
        ]
    )
    expected = linear_cell(rel, num, periodic)
    got = []
    for r in rel:
        r = periodic_position(float(r), num) if periodic else float(r)
        left = left_node(r, num, periodic)
        got.append((left, right_node(left, num, periodic), cell_fraction(r, left)))
    left, right, fraction = (np.array(column) for column in zip(*got, strict=True))
    np.testing.assert_array_equal(left, expected[0])
    np.testing.assert_array_equal(right, expected[1])
    np.testing.assert_allclose(fraction, expected[2], rtol=0, atol=1e-12)
