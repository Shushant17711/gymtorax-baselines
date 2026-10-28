"""Resumable sweep: queue of (method x seed) training runs, packed onto CPU slots.

  uv run python scripts/sweep.py configs/sweep/main.yaml --dry-run    # list jobs and their status
  uv run python scripts/sweep.py configs/sweep/main.yaml              # run everything not finished
  uv run python scripts/sweep.py configs/sweep/main.yaml --only ppo   # one method

Each job is `scripts/train.py <config> seed=<s> <overrides>` in its own process, logging to logs/sweep/.
A job is finished when its run dir has eval.json, so killing and relaunching the sweep resumes it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

from gtb.config import load_config
from gtb.train import run_dir_for

BASE = "configs/base.yaml"


def jobs_for(sweep: dict, root: str = "results", only: list[str] | None = None) -> list[dict]:
    out = []
    for m, spec in sweep["methods"].items():
        if only and m not in only:
            continue
        for s in sweep["seeds"]:
            ov = [*spec.get("overrides", []), f"seed={s}"]
            rd = run_dir_for(load_config([BASE, spec["config"]], ov), root)
            out.append({"method": m, "seed": s, "config": spec["config"], "overrides": ov,
                        "slots": int(spec.get("slots", 1)), "run_dir": rd, "done": (rd / "eval.json").exists()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--root", default="results")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu-slots", type=int)
    a = ap.parse_args()
    sweep = yaml.safe_load(Path(a.sweep).read_text())
    jobs = jobs_for(sweep, a.root, a.only)
    todo = [j for j in jobs if not j["done"]]
    for j in jobs:
        print(f"{'done' if j['done'] else 'todo'}  {j['method']:7s} seed{j['seed']}  {j['run_dir']}")
    print(f"{len(jobs)} jobs, {len(todo)} to run")
    if a.dry_run or not todo:
        return
    cap = a.cpu_slots or int(sweep.get("cpu_slots", 20))
    logs = Path("logs/sweep")
    logs.mkdir(parents=True, exist_ok=True)
    running: list[tuple[subprocess.Popen, dict]] = []
    failed = []
    while todo or running:
        for p, j in list(running):
            if p.poll() is not None:
                running.remove((p, j))
                ok = p.returncode == 0 and (j["run_dir"] / "eval.json").exists()
                print(f"[{time.strftime('%H:%M:%S')}] {'finished' if ok else 'FAILED'} {j['method']} seed{j['seed']}",
                      flush=True)
                if not ok:
                    failed.append(j)
        used = sum(j["slots"] for _, j in running)
        for j in list(todo):
            if used + min(j["slots"], cap) <= cap:
                todo.remove(j)
                log = (logs / f"{j['method']}_seed{j['seed']}.log").open("w")
                cmd = [sys.executable, "scripts/train.py", j["config"], *j["overrides"], "--root", a.root]
                running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), j))
                used += min(j["slots"], cap)
                print(f"[{time.strftime('%H:%M:%S')}] started {j['method']} seed{j['seed']} ({j['slots']} slots)",
                      flush=True)
        time.sleep(5)
    print(f"sweep done; {len(failed)} failed: {[(j['method'], j['seed']) for j in failed]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
