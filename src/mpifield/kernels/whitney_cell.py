"""The edge (Whitney) gather and path deposit in any dimension, vectorized: the NumPy kernels.

The NumPy versions of the ``gather_edges_*`` and ``deposit_paths_*`` kernels call these,
with the same arithmetic as their Pyccel versions (``whitney_cell_pyccel.py``) and CUDA
versions (``whitney_cell.cuh``). ``edges`` is ``num_nodes + (ndim,)``: ``edges[..., a]``
the line integral along the edge from a node to its neighbour along axis ``a``;
``geometry`` is ``(*lower, *cell_size, *num_nodes, *periodic)``.
"""

from __future__ import annotations

import itertools

import numpy as np

from mpifield.kernels.linear_cell import linear_cell


def _split(geometry: tuple) -> tuple[np.ndarray, np.ndarray, list[int], list[bool]]:
    """``geometry`` as lower, cell size, node counts and periodic flags."""
    ndim = len(geometry) // 4
    lower = np.asarray(geometry[:ndim], dtype=float)
    size = np.asarray(geometry[ndim : 2 * ndim], dtype=float)
    nodes = [int(n) for n in geometry[2 * ndim : 3 * ndim]]
    periodic = [bool(p) for p in geometry[3 * ndim :]]
    return lower, size, nodes, periodic


def _cell_edges(
    cells: list[tuple[np.ndarray, np.ndarray, np.ndarray]], axis: int
) -> list[tuple[tuple[np.ndarray, ...], np.ndarray, tuple[int, ...]]]:
    """The edges of the cells along ``axis`` and their weights (the linear ones across).

    ``cells`` holds ``(left, right, fraction)`` per axis. Returns per edge the node
    index of its start, its weight and whether it is the upper one along each other
    axis: ``2**(ndim - 1)`` edges.
    """
    ndim = len(cells)
    others = [other for other in range(ndim) if other != axis]
    result = []
    for bits in itertools.product((0, 1), repeat=len(others)):
        index = [cells[axis][0]] * ndim
        weight = np.ones_like(cells[axis][2])
        for other, upper in zip(others, bits, strict=True):
            left, right, fraction = cells[other]
            index[other] = right if upper else left
            weight = weight * (fraction if upper else 1.0 - fraction)
        result.append((tuple(index), weight, bits))
    return result


def gather_edges(
    positions: np.ndarray, edges: np.ndarray, result: np.ndarray, geometry: tuple
) -> None:
    """The edge field of ``edges`` at ``positions`` ``(ndim, N)`` into ``result`` ``(ndim, N)``."""
    lower, size, nodes, periodic = _split(geometry)
    ndim = lower.size
    cells = [
        linear_cell(
            (positions[axis] - lower[axis]) / size[axis], nodes[axis], periodic[axis]
        )
        for axis in range(ndim)
    ]
    for axis in range(ndim):
        total = np.zeros(positions.shape[1])
        for index, weight, _ in _cell_edges(cells, axis):
            total += weight * edges[(*index, axis)]
        result[axis] = total / size[axis]


def _pieces(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The paths from ``a`` to ``b`` (in cells, ``(ndim, N)``) cut at every grid line.

    Returns the path of every piece and its two ends ``(ndim, pieces)``, in the order
    of the paths and along each path.
    """
    ndim, count = a.shape
    step = b - a
    paths, times = [np.arange(count)] * 2, [np.zeros(count), np.ones(count)]
    for axis in range(ndim):
        low = np.minimum(a[axis], b[axis])
        high = np.maximum(a[axis], b[axis])
        first = np.floor(low).astype(np.int64) + 1
        last = np.ceil(high).astype(np.int64) - 1
        crossings = np.maximum(last - first + 1, 0)
        path = np.repeat(np.arange(count), crossings)
        offset = np.repeat(np.cumsum(crossings) - crossings, crossings)
        line = first[path] + np.arange(path.size) - offset
        paths.append(path)
        times.append((line - a[axis, path]) / step[axis, path])
    path = np.concatenate(paths)
    time = np.concatenate(times)
    order = np.lexsort((time, path))
    path, time = path[order], time[order]
    same = path[1:] == path[:-1]
    path, begin, finish = path[1:][same], time[:-1][same], time[1:][same]

    def at(t: np.ndarray) -> np.ndarray:
        # the end of a path is its end, not start + 1 * step
        return np.where(t >= 1.0, b[:, path], a[:, path] + t * step[:, path])

    return path, at(begin), at(finish)


def deposit_paths(
    start: np.ndarray,
    end: np.ndarray,
    charges: np.ndarray,
    edges: np.ndarray,
    geometry: tuple,
) -> None:
    """Add the current of the paths from ``start`` to ``end`` ``(ndim, N)`` to ``edges``.

    Each path is cut at the grid lines; on a bounded axis its pieces are clamped to the
    grid. A piece adds ``q`` times its length along an axis (in cells) to the cell's
    edges along that axis, times the average of their weights over the piece.
    """
    lower, size, nodes, periodic = _split(geometry)
    ndim = lower.size
    a = (start - lower[:, None]) / size[:, None]
    b = (end - lower[:, None]) / size[:, None]
    keep = (charges != 0.0) & np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
    a, b, charges = a[:, keep], b[:, keep], charges[keep]
    path, x0, x1 = _pieces(a, b)
    for axis in range(ndim):
        if not periodic[axis]:
            x0[axis] = np.clip(x0[axis], 0.0, nodes[axis] - 1.0)
            x1[axis] = np.clip(x1[axis], 0.0, nodes[axis] - 1.0)
    middle = 0.5 * (x0 + x1)
    delta = x1 - x0
    cells = [
        linear_cell(middle[axis], nodes[axis], periodic[axis]) for axis in range(ndim)
    ]
    q = charges[path]
    for axis in range(ndim):
        others = [other for other in range(ndim) if other != axis]
        for index, weight, bits in _cell_edges(cells, axis):
            if ndim == 3:
                # the weight is quadratic along the piece: its average is the
                # midpoint's plus or minus delta_u delta_w / 12
                sign = 1.0 if bits[0] == bits[1] else -1.0
                weight = weight + sign * delta[others[0]] * delta[others[1]] / 12.0
            np.add.at(edges, (*index, axis), weight * q * delta[axis])
