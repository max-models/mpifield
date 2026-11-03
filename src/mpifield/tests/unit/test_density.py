"""Tests for the density deposit and the node volumes."""

from __future__ import annotations

import numpy as np
import pytest

from mpifield import LINEAR, QUADRATIC, Field


def _field(periodic: tuple[bool, ...], num_components: int = 1) -> Field:
    return Field(
        ndim=2,
        lower_bound=(0.0, 0.0),
        upper_bound=(1.0, 1.0),
        num_gridpoints=(11, 9),
        num_components=num_components,
        periodic=periodic,
    )


@pytest.mark.parametrize("weighting", [LINEAR, QUADRATIC], ids=lambda w: w.name)
@pytest.mark.parametrize("periodic", [(False, False), (False, True), (True, True)])
def test_the_density_times_the_node_volumes_is_the_total(weighting, periodic) -> None:
    """Nothing is lost, also at the edges where a node collects half a cell."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0.0, 1.0, (2, 5000))
    values = rng.normal(0.0, 1.0, (3, 5000))
    field = _field(periodic, 3)
    field.deposit_density(positions, values, weighting)
    totals = (field.to_numpy() * field.node_volumes().to_numpy()).sum(axis=(0, 1))
    np.testing.assert_allclose(totals, values.sum(axis=1), rtol=1e-12, atol=1e-12)


def test_a_uniform_cloud_has_a_uniform_density_up_to_the_edges() -> None:
    """The half volumes make the boundary nodes as dense as the inner ones."""
    count = 200_000
    positions = np.random.default_rng(1).uniform(0.0, 1.0, (2, count))
    field = _field((False, False))
    field.deposit_density(positions, np.full(count, 1.0 / count))
    density = field.to_numpy()[..., 0]
    assert density.mean() == pytest.approx(1.0, abs=0.02)
    assert density[0].mean() == pytest.approx(1.0, abs=0.05)
    assert density[:, -1].mean() == pytest.approx(1.0, abs=0.05)


def test_deposit_density_adds_to_the_field() -> None:
    field = _field((True, True))
    field.set_field_data(2.0)
    field.deposit_density(np.full((2, 1), 0.5), 0.0)
    np.testing.assert_allclose(field.to_numpy(), 2.0)
    density = _field((True, True))
    density.deposit_density(np.full((2, 1), 0.5), 1.0)
    field.deposit_density(np.full((2, 1), 0.5), 1.0)
    np.testing.assert_allclose(field.to_numpy(), 2.0 + density.to_numpy())


def test_node_volumes_halve_on_bounded_edges() -> None:
    volumes = _field((False, True)).node_volumes().to_numpy()[..., 0]
    cell = 0.1 * (1.0 / 9)
    assert volumes.shape == (11, 9)
    np.testing.assert_allclose(volumes[1:-1], cell)
    np.testing.assert_allclose(volumes[[0, -1]], cell / 2)
    assert volumes.sum() == pytest.approx(1.0)
