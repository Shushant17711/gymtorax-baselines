"""Tune the classical (feed-forward + PI q_min) controller with Optuna TPE on training scenarios.

Objective: mean return over a fixed set of training-split scenarios (common random numbers across trials).
Trial 0 is the bundled scripted schedule. Writes configs/agent/classical.yaml and results/tuning/classical.jsonl.
"""
import argparse
import json
import sys
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


def scripted_value(trials, include_waiting: bool = False):
    """Score of the completed trial that ran the scripted parameters (or True if one is queued)."""
    for t in trials:
        same = t.params == SCRIPTED_PARAMS or (include_waiting and t.system_attrs.get("fixed_params") == SCRIPTED_PARAMS)
        if same and t.value is not None:
            return t.value
        if same and include_waiting and t.state == optuna.trial.TrialState.WAITING:
            return True
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--n-variants", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="configs/agent/classical.yaml")
    ap.add_argument("--q-cap", type=float, default=None, help="cap Q in the reward (amendment 1)")
    args = ap.parse_args()

    rng = np.random.default_rng(1000 + args.seed)
    variants = [sample_variant(rng, "train") for _ in range(args.n_variants)]
    from gtb.envs.wrapper import set_q_cap
    from gtb.train import EXIT_PAUSED, pause_file

    set_q_cap(args.q_cap)  # before the worker pool is spawned
    name = Path(args.out).stem
    log = Path("results/tuning") / (name + ".jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    # The study lives in SQLite, so tuning resumes after a pause or crash with every finished trial kept.
    storage = f"sqlite:///{Path('results/tuning') / (name + '.db')}"
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed),
                                study_name=name, storage=storage, load_if_exists=True)
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)  # interrupted mid-batch: redone as new trials
    complete = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not study.trials:
        log.write_text("")
    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.WAITING)
    if scripted_value(study.get_trials(deepcopy=False, states=states), include_waiting=True) is None:
        study.enqueue_trial(dict(SCRIPTED_PARAMS))  # trial 0 = the bundled schedule (re-queued if it was cut off)
    elif complete:
        print(f"resuming tuning: {len(complete)}/{args.trials} trials done", flush=True)
    t0 = time.time()
    with make_pool(args.workers) as pool:
        done = len(complete)
        while done < args.trials:
            if pause_file().exists():
                print(f"paused after {done} trials", flush=True)
                sys.exit(EXIT_PAUSED)
            trials = [study.ask() for _ in range(min(args.batch, args.trials - done))]
            params = [suggest(t) for t in trials]
            jobs = [(partial(classical_policy, p), v, 0, None) for p in params for v in variants]
            recs = run_jobs(jobs, pool=pool)
            for i, (t, p) in enumerate(zip(trials, params, strict=True)):
                rs = recs[i * len(variants):(i + 1) * len(variants)]
                score = float(np.mean([r["return"] for r in rs]))
                if any(r["failed"] for r in rs):
                    score -= 1.0
                t.set_user_attr("env_steps", int(sum(r["steps"] for r in rs)))
                study.tell(t, score)
                with log.open("a") as f:
                    f.write(json.dumps({"trial": t.number, "score": score, "params": p,
                                        "failed": any(r["failed"] for r in rs)}) + "\n")
            done += len(trials)
            print(f"[{time.time()-t0:6.0f}s] trials {done}/{args.trials} best {study.best_value:.4f} "
                  f"(scripted {scripted_value(study.trials) or float("nan"):.4f})", flush=True)
    complete = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    env_steps = int(sum(t.user_attrs.get("env_steps", 0) for t in study.trials))
    out = {"name": "classical", "params": study.best_params, "q_cap": args.q_cap, "tuning": {
        "trials": args.trials, "variants": variants, "env_steps": env_steps, "best_train_score": study.best_value,
        "scripted_train_score": scripted_value(complete), "seed": args.seed}}
    Path(args.out).write_text(yaml.safe_dump(out, sort_keys=False))
    print(json.dumps(study.best_params, indent=1))


if __name__ == "__main__":
    main()
