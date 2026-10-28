"""Evaluate a trained run or a reference controller on the frozen scenario set.

  uv run python scripts/evaluate.py --run results/ppo/<hash>/seed0          # rewrites <run>/eval.json
  uv run python scripts/evaluate.py --reference scripted                     # results/reference/scripted/eval.json
  uv run python scripts/evaluate.py --reference classical --params configs/agent/classical.yaml
"""
import argparse
import json
from functools import partial
from pathlib import Path

import yaml

from gtb.agents import reference
from gtb.config import load_config
from gtb.envs.variants import load_eval_set
from gtb.eval.rollout import eval_suite
from gtb.train import evaluate_run, summarise


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--reference", choices=["random", "scripted", "classical"])
    ap.add_argument("--params", help="YAML with a `params:` dict (classical)")
    ap.add_argument("--seeds", type=int, default=5, help="random policy: seeds per scenario")
    ap.add_argument("--eval-set", default="configs/env/heldout.yaml")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--q-cap", type=float, default=None, help="reference only: cap Q in the reward (amendment 1)")
    ap.add_argument("--root", default="results", help="reference only: results root")
    a = ap.parse_args()
    from gtb.envs.wrapper import set_q_cap

    set_q_cap(a.q_cap)

    if a.run:
        rd = Path(a.run)
        ev = evaluate_run(rd, load_config(rd / "config.yaml"), workers=a.workers)
        out = rd / "eval.json"
        old = json.loads(out.read_text()) if out.exists() else {}
        ev = {k: old[k] for k in ("env_steps", "wall_s") if k in old} | ev
    else:
        es = load_eval_set(a.eval_set)
        if a.reference == "random":
            suites = [eval_suite(partial(reference.random_policy, s), es, workers=a.workers) for s in range(a.seeds)]
            suite = {k: [r for su in suites for r in su[k]] for k in es}
        elif a.reference == "scripted":
            suite = eval_suite(reference.scripted_policy, es, workers=a.workers)
        else:
            params = yaml.safe_load(Path(a.params).read_text())["params"] if a.params else None
            suite = eval_suite(partial(reference.classical_policy, params), es, workers=a.workers)
        ev = {"env_steps": 0, "summary": summarise(suite), "episodes": suite}
        out = Path(a.root) / "reference" / a.reference / "eval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, default=float))
    print(json.dumps(ev["summary"], indent=1))


if __name__ == "__main__":
    main()
