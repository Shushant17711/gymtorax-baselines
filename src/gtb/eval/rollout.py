"""Run policies on lists of scenarios, in parallel worker processes.

A *policy* has `reset(env)` and `act(obs, env) -> np.ndarray in [-1,1]^7`.
`policy_factory` must be picklable (top-level function or functools.partial) and is called inside the worker.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

COMPONENTS = ("fusion_gain", "h98", "q_min", "q_edge")


def run_episode(env, policy, variant: dict, seed: int = 0, record: bool = False) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed, options={"variant": variant})
    policy.reset(env)
    ret, n, done, failed = 0.0, 0, False, False
    comps = dict.fromkeys(COMPONENTS, 0.0)
    viol_qmin = viol_q95 = 0
    traj = {"obs": [obs], "act": [], "rew": []} if record else None
    while not done:
        a = np.clip(np.asarray(policy.act(obs, env), np.float64), -1, 1)
        obs, r, te, tr, info = env.step(a)
        ret += r
        n += 1
        for k in COMPONENTS:
            comps[k] += info["components"][k]
        viol_qmin += bool(info["viol_q_min"])
        viol_q95 += bool(info["viol_q95"])
        failed |= bool(info["failed"])
        done = te or tr
        if record:
            traj["obs"].append(obs)
            traj["act"].append(a)
            traj["rew"].append(r)
    out = {"return": ret, "steps": n, "failed": failed, "variant": dict(variant), "seed": seed,
           "frac_q_min_lt_1": viol_qmin / max(n, 1), "frac_q95_lt_3": viol_q95 / max(n, 1),
           **{f"ret_{k}": v for k, v in comps.items()}}
    if record:
        out["traj"] = {k: np.asarray(v) for k, v in traj.items()}
    return out


_ENV = None


def _worker(args):
    global _ENV
    policy_factory, variant, seed, norm_path = args
    if _ENV is None:
        from gtb.envs.wrapper import FlatEnv, load_norm

        _ENV = FlatEnv(norm=load_norm(norm_path) if norm_path else None)
    return run_episode(_ENV, policy_factory(), variant, seed)


def make_pool(workers: int | None = None) -> ProcessPoolExecutor:
    """Persistent spawn pool; workers keep their env (and JIT caches) across calls."""
    import multiprocessing as mp

    return ProcessPoolExecutor(workers or os.cpu_count() or 1, mp_context=mp.get_context("spawn"))


def evaluate(policy_factory: Callable[[], Any], variants: list[dict], seeds: list[int] | None = None,
             workers: int | None = None, norm_path: str | None = "configs/env/norm.json",
             pool: ProcessPoolExecutor | None = None) -> list[dict]:
    """One episode per (variant, seed). Returns records in input order."""
    return run_jobs([(policy_factory, v, s, norm_path) for v in variants for s in (seeds or [0])],
                    workers=workers, pool=pool)


def run_jobs(jobs: list[tuple], workers: int | None = None, pool: ProcessPoolExecutor | None = None) -> list[dict]:
    """jobs: (policy_factory, variant, seed, norm_path) tuples."""
    if pool is not None:
        return list(pool.map(_worker, jobs))
    workers = min(workers or os.cpu_count() or 1, len(jobs))
    if workers <= 1:
        return [_worker(j) for j in jobs]
    with make_pool(workers) as ex:
        return list(ex.map(_worker, jobs))


def eval_suite(policy_factory, eval_set: dict[str, list[dict]], workers: int | None = None,
               norm_path: str | None = "configs/env/norm.json",
               pool: ProcessPoolExecutor | None = None) -> dict[str, list[dict]]:
    """Evaluate on every split of an eval set (nominal/heldout/ood) in a single pool."""
    flat = [(split, v) for split, vs in eval_set.items() for v in vs]
    recs = evaluate(policy_factory, [v for _, v in flat], workers=workers, norm_path=norm_path, pool=pool)
    out: dict[str, list[dict]] = {k: [] for k in eval_set}
    for (split, _), r in zip(flat, recs, strict=True):
        out[split].append(r)
    return out
