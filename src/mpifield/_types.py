"""Typing aliases and the small value types shared by mpifield's modules."""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias

from numpy import ndarray

Array: TypeAlias = ndarray
Number: TypeAlias = int | float
PathLikeStr: TypeAlias = str | Path | PathLike[str]
IndexLike: TypeAlias = int | slice | tuple[int | slice, ...]
Kernel: TypeAlias = Callable[..., Any]


class Bounds(NamedTuple):
    """The lower and upper corner of a grid block, one entry per axis.

    The kernel wrappers (:meth:`~mpifield.Field.interpolate_field_2d`, ...) only read
    ``lower_bound``, so any object with that attribute will do.
    """

    lower_bound: Array
    upper_bound: Array
