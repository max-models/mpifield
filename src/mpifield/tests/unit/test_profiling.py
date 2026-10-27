"""The opt-in scope-profiler regions: off unless the application enables the tag."""

from __future__ import annotations

import numpy as np
import pytest
import scope_profiler as sp
from maybempi import MPI

from mpifield import Field

# the profiler gathers its results on rank 0 under MPI
pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1, reason="serial: results are gathered on rank 0"
)


@pytest.fixture(autouse=True)
def _fresh_profiler(monkeypatch):
    """Keep the profiling state of these tests out of the others."""
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)
    sp.reset()
    yield
    sp.reset()


def _region_names(**setup_kwargs) -> set[str]:
    sp.setup(output=None, **setup_kwargs)
    # an application region, recorded either way
    field = Field(ndim=1, lower_bound=(0.0,), upper_bound=(1.0,), num_gridpoints=5)
    with sp.region("user.run"):
        field.scatter(np.array([[0.3]]), 1.0)
    results = sp.finalize(verbose=False, return_results=True)
    # "*" also enables the regions of the libraries below (cunumpy's kernels)
    return {
        name for name in results.region_names if name.startswith(("mpifield.", "user."))
    }


def test_regions_are_off_by_default() -> None:
    assert _region_names() == {"user.run"}


@pytest.mark.parametrize("tags", [["mpifield"], ["*"]])
def test_enabling_the_tag_records_the_regions(tags: list[str]) -> None:
    assert _region_names(enable_tags=tags) == {
        "mpifield.Field.scatter",
        "user.run",
    }
