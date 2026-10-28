import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "scripts")
from sweep import jobs_for


def test_main_sweep_lists_20_distinct_jobs():
    jobs = jobs_for(yaml.safe_load(Path("configs/sweep/main.yaml").read_text()), root="/nonexistent")
    assert len(jobs) == 20 and len({str(j["run_dir"]) for j in jobs}) == 20
    assert {j["method"] for j in jobs} == {"ppo", "sac", "mbrl", "pimbrl"}
    assert not any(j["done"] for j in jobs)


def test_resume_skips_finished(tmp_path):
    sw = yaml.safe_load(Path("configs/sweep/main.yaml").read_text())
    j = jobs_for(sw, root=str(tmp_path))[0]
    rd = j["run_dir"]
    rd.mkdir(parents=True)
    (rd / "eval.json").write_text(json.dumps({}))
    jobs = jobs_for(sw, root=str(tmp_path))
    assert sum(x["done"] for x in jobs) == 1
    out = subprocess.run([sys.executable, "scripts/sweep.py", "configs/sweep/main.yaml", "--root", str(tmp_path),
                          "--dry-run"], capture_output=True, text=True, check=True).stdout
    assert "20 jobs, 19 to run" in out
