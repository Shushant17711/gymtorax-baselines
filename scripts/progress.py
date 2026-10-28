"""Print progress of every run in both sweeps (used by `scripts/gtb status`)."""
import json
from pathlib import Path

import yaml
from sweep import jobs_for


def steps_of(rd: Path) -> int:
    m = rd / "metrics.jsonl"
    if not m.exists() or m.stat().st_size == 0:
        return 0
    with m.open("rb") as f:
        f.seek(max(0, m.stat().st_size - 4096))
        last = [x for x in f.read().decode(errors="ignore").splitlines() if x.startswith("{")]
    try:
        return int(json.loads(last[-1])["steps"]) if last else 0
    except (json.JSONDecodeError, KeyError):
        return 0


for name, sweep, root in (("original reward", "configs/sweep/main.yaml", "results"),
                          ("capped reward", "configs/sweep/capped.yaml", "results/capped")):
    jobs = jobs_for(yaml.safe_load(Path(sweep).read_text()), root)
    print(f"\n{name}: {sum(j['done'] for j in jobs)}/{len(jobs)} runs finished")
    for j in jobs:
        rd, budget = j["run_dir"], int(next(o for o in j["overrides"] if o.startswith("train.steps=")).split("=")[1])
        if j["done"]:
            state = "done"
        elif (rd / "ckpt" / "progress.json").exists() or steps_of(rd):
            ck = rd / "ckpt" / "progress.json"
            saved = f"checkpoint at {json.loads(ck.read_text())['steps']:,}" if ck.exists() else "no checkpoint yet"
            state = f"{steps_of(rd):>7,}/{budget:,} steps  ({saved})"
        else:
            continue
        print(f"  {j['method']:7s} seed{j['seed']}  {state}")
