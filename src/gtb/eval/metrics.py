"""Aggregate metrics (Agarwal et al. 2021): IQM and stratified-bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from scipy import stats


def iqm(x: np.ndarray) -> float:
    """Interquartile mean over all entries (25% trimmed mean)."""
    return float(stats.trim_mean(np.asarray(x, float).reshape(-1), 0.25))


def stratified_bootstrap_ci(scores: np.ndarray, stat=iqm, n_boot: int = 2000, alpha: float = 0.05,
                            seed: int = 0) -> tuple[float, float, float]:
    """scores: (n_runs, n_tasks). Resample runs independently within each task (column).

    Returns (point, lo, hi).
    """
    s = np.asarray(scores, float)
    if s.ndim == 1:
        s = s[:, None]
    rng = np.random.default_rng(seed)
    n, m = s.shape
    idx = rng.integers(0, n, size=(n_boot, n, m))
    boot = np.array([stat(np.take_along_axis(s, idx[b], axis=0)) for b in range(n_boot)])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return stat(s), float(lo), float(hi)


def prob_improvement(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) averaged over tasks; x, y: (n_runs, n_tasks). Ties count 1/2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = []
    for j in range(x.shape[1]):
        gt = (x[:, j][:, None] > y[:, j][None, :]).mean()
        eq = (x[:, j][:, None] == y[:, j][None, :]).mean()
        out.append(gt + 0.5 * eq)
    return float(np.mean(out))


def iqm_diff_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                seed: int = 0) -> tuple[float, float, float]:
    """IQM(x) - IQM(y) with a stratified-bootstrap CI.

    x, y: (n_seeds, n_scenarios) returns, same scenario columns. Seeds are resampled independently per method
    within each scenario (the seeds of the two methods are unrelated); scenarios are fixed strata.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must share scenario columns")
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, x.shape[0], size=x.shape)
        iy = rng.integers(0, y.shape[0], size=y.shape)
        d[b] = iqm(np.take_along_axis(x, ix, 0)) - iqm(np.take_along_axis(y, iy, 0))
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return iqm(x) - iqm(y), float(lo), float(hi)
