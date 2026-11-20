"""Iterative proportional fitting (raking) on a multidimensional table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Margin:
    axes: tuple[int, ...]  # axes of the joint table kept in the margin, ascending
    target: np.ndarray  # shape = shape of the joint table on these axes; NaN = unconstrained cell

    def __post_init__(self):
        if list(self.axes) != sorted(set(self.axes)):
            raise ValueError(f"margin axes must be ascending without repeats: {self.axes}")


@dataclass(frozen=True)
class IPFResult:
    table: np.ndarray
    iterations: int
    max_abs_error: float
    converged: bool


def _project(table: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    drop = tuple(i for i in range(table.ndim) if i not in axes)
    return table.sum(axis=drop)


def ipf(
    seed: np.ndarray,
    margins: list[Margin],
    *,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> IPFResult:
    """Fits `seed` to the given marginal tables.

    Zero cells in `seed` stay zero (structural zeros). The marginal tables must agree
    on shared axes, otherwise the algorithm will not converge. A margin cell with NaN
    is not constrained (partial margin, e.g. only for KRS forms).
    """
    table = seed.astype(float).copy()
    for m in margins:
        expected = tuple(table.shape[a] for a in m.axes)
        if m.target.shape != expected:
            raise ValueError(f"margin on axes {m.axes}: shape {m.target.shape} ≠ {expected}")
    err = np.inf
    for it in range(1, max_iter + 1):
        for m in margins:
            current = _project(table, m.axes)
            known = ~np.isnan(m.target)
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(known, np.where(current > 0, m.target / current, 0.0), 1.0)
            shape = [table.shape[i] if i in m.axes else 1 for i in range(table.ndim)]
            table *= factor.reshape(shape)
        err = max(float(np.nanmax(np.abs(_project(table, m.axes) - m.target))) for m in margins)
        if err < tol:
            return IPFResult(table, it, err, True)
    return IPFResult(table, max_iter, err, False)
