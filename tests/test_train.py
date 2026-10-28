import json

import pytest

from gtb.config import load_config
from gtb.train import run_dir_for


def test_run_dir_excludes_seed():
    a = load_config(["configs/base.yaml", "configs/agent/dummy.yaml"], ["seed=1"])
    b = load_config(["configs/base.yaml", "configs/agent/dummy.yaml"], ["seed=2"])
    assert run_dir_for(a).parent == run_dir_for(b).parent and run_dir_for(a).name == "seed1"


@pytest.mark.slow
def test_dummy_smoke_run(tmp_path):
    from gtb.train import train

    cfg = load_config(["configs/base.yaml", "configs/agent/dummy.yaml"],
                      ["train.steps=300", "train.eval_every=200", "train.periodic_splits=[nominal]",
                       "train.eval_workers=4"])
    rd = train(cfg, root=str(tmp_path))
    for f in ("config.yaml", "metrics.jsonl", "model.json", "eval.json"):
        assert (rd / f).exists(), f
    lines = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    assert any("episode_return" in x for x in lines)
    assert sum("eval" in x for x in lines) == 2  # one periodic (at 200) + final
    ev = json.loads((rd / "eval.json").read_text())
    assert ev["env_steps"] == 300 and set(ev["summary"]) == {"nominal", "heldout", "ood"}
    assert ev["summary"]["heldout"]["n"] == 15
    assert train(cfg, root=str(tmp_path)) == rd  # second call is a no-op


@pytest.mark.slow
@pytest.mark.parametrize("algo,steps,extra", [
    ("ppo", 600, ["agent.n_envs=2", "agent.n_steps=150", "agent.batch_size=100", "agent.n_epochs=2"]),
    ("sac", 400, ["agent.learning_starts=200", "agent.batch_size=64"]),
])
def test_sb3_smoke(tmp_path, algo, steps, extra):
    from gtb.train import train

    cfg = load_config(["configs/base.yaml", f"configs/agent/{algo}.yaml"],
                      [f"train.steps={steps}", "train.eval_every=100000", "train.eval_workers=8", *extra])
    rd = train(cfg, root=str(tmp_path))
    assert (rd / "model.zip").exists()
    ev = json.loads((rd / "eval.json").read_text())
    assert ev["env_steps"] >= steps and ev["summary"]["ood"]["n"] == 5
    assert all(r["steps"] > 0 for r in ev["episodes"]["nominal"])
    if algo == "ppo":  # evaluate.py reloads the saved model and reproduces the deterministic eval
        import subprocess
        import sys

        before = ev["summary"]["nominal"]["mean"]
        subprocess.run([sys.executable, "scripts/evaluate.py", "--run", str(rd), "--workers", "8"], check=True,
                       capture_output=True)
        assert json.loads((rd / "eval.json").read_text())["summary"]["nominal"]["mean"] == pytest.approx(before)
