"""Tune the classical (feed-forward + PI q_min) controller with Optuna TPE on training scenarios.

Objective: mean return over a fixed set of training-split scenarios (common random numbers across trials).
Trial 0 is the bundled scripted schedule. Writes configs/agent/classical.yaml and results/tuning/classical.jsonl.
"""
import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import yaml

from gtb.agents.reference import PARAM_RANGES, SCRIPTED_PARAMS, classical_policy
from gtb.envs.variants import sample_variant
from gtb.eval.rollout import make_pool, run_jobs

LOG_PARAMS = set()


def suggest(trial):
    return {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in PARAM_RANGES.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--n-variants", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="configs/agent/classical.yaml")
    args = ap.parse_args()

    rng = np.random.default_rng(1000 + args.seed)
    variants = [sample_variant(rng, "train") for _ in range(args.n_variants)]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.enqueue_trial(dict(SCRIPTED_PARAMS))
    log = Path("results/tuning/classical.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    env_steps, t0 = 0, time.time()
    with make_pool(args.workers) as pool:
        done = 0
        while done < args.trials:
            trials = [study.ask() for _ in range(min(args.batch, args.trials - done))]
            params = [suggest(t) for t in trials]
            jobs = [(partial(classical_policy, p), v, 0, None) for p in params for v in variants]
            recs = run_jobs(jobs, pool=pool)
            for i, (t, p) in enumerate(zip(trials, params, strict=True)):
                rs = recs[i * len(variants):(i + 1) * len(variants)]
                score = float(np.mean([r["return"] for r in rs]))
                if any(r["failed"] for r in rs):
                    score -= 1.0
                study.tell(t, score)
                env_steps += sum(r["steps"] for r in rs)
                with log.open("a") as f:
                    f.write(json.dumps({"trial": t.number, "score": score, "params": p,
                                        "failed": any(r["failed"] for r in rs)}) + "\n")
            done += len(trials)
            print(f"[{time.time()-t0:6.0f}s] trials {done}/{args.trials} best {study.best_value:.4f} "
                  f"(scripted {study.trials[0].value:.4f})", flush=True)
    out = {"name": "classical", "params": study.best_params, "tuning": {
        "trials": args.trials, "variants": variants, "env_steps": env_steps, "best_train_score": study.best_value,
        "scripted_train_score": study.trials[0].value, "seed": args.seed}}
    Path(args.out).write_text(yaml.safe_dump(out, sort_keys=False))
    print(json.dumps(study.best_params, indent=1))


if __name__ == "__main__":
    main()
