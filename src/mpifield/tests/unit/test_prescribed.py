"""Tests for electric fields from a prescribed potential."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from mpifield import axial_potential, electric_field_from_potential


def test_uniform_field_from_a_linear_potential() -> None:
    """phi falling 15 V over 0.2 m gives E = +75 V/m everywhere, ends included."""
    field = electric_field_from_potential(
        axial_potential([0.0, 0.2], [15.0, 0.0]), (0.0,), (0.2,), 21
    )
    assert np.allclose(field.field_data.local_with_halos[..., 0], 75.0)


def test_field_from_a_smooth_potential_matches_the_derivative() -> None:
    """E = -dphi/dx for phi = sin(kx) away from the one-sided end points."""
    k = 3.0

    def potential(x: xp.ndarray) -> xp.ndarray:
        return xp.sin(k * x)

    field = electric_field_from_potential(potential, (0.0,), (1.0,), 401)
    x = np.linspace(0.0, 1.0, 401)
    expected = -k * np.cos(k * x)
    assert np.allclose(
        field.field_data.local_with_halos[5:-5, 0], expected[5:-5], atol=1e-3
    )


def test_two_dimensional_field_has_a_component_per_axis() -> None:
    """phi = 2x + 3y gives E = (-2, -3) and a two-component field."""

    def potential(x: xp.ndarray, y: xp.ndarray) -> xp.ndarray:
        return 2.0 * x + 3.0 * y

    field = electric_field_from_potential(potential, (0.0, 0.0), (1.0, 2.0), (9, 11))
    data = field.field_data.local_with_halos
    assert data.shape[-1] == 2
    assert np.allclose(data[..., 0], -2.0)
    assert np.allclose(data[..., 1], -3.0)


def test_axial_potential_uses_the_chosen_axis() -> None:
    """The profile depends on one axis only and is clamped outside its table."""
    potential = axial_potential([0.0, 1.0], [0.0, 10.0], axis=1)
    x, y = np.meshgrid([0.0, 5.0], [-1.0, 0.5, 2.0], indexing="ij")
    assert np.allclose(potential(x, y), [[0.0, 5.0, 10.0]] * 2)


def test_bad_potential_tables_are_rejected() -> None:
    """Mismatched or non-increasing tables raise."""
    with pytest.raises(ValueError):
        axial_potential([0.0, 1.0], [1.0])
    with pytest.raises(ValueError):
        axial_potential([1.0, 0.0], [1.0, 2.0])
