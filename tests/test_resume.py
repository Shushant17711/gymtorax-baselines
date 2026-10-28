"""Pause / resume: checkpoints are atomic, resumed runs finish with the exact step budget, and MBRL resumes
bit-for-bit (its whole state is checkpointed)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gtb.train import EXIT_PAUSED, _ckpt_prepare


def _mk(d: Path, complete: bool):
    d.mkdir()
    (d / "x").write_text(d.name)
    if complete:
        (d / "progress.json").write_text("{}")


def test_ckpt_prepare_recovers_every_crash_point(tmp_path):
    # crash while writing ckpt.new: keep the old checkpoint
    _mk(tmp_path / "ckpt", True)
    _mk(tmp_path / "ckpt.new", False)
    assert (_ckpt_prepare(tmp_path) / "x").read_text() == "ckpt"
    assert not (tmp_path / "ckpt.new").exists()
    # crash after ckpt -> ckpt.old, before ckpt.new -> ckpt: the new one is complete, use it
    (tmp_path / "ckpt").rename(tmp_path / "ckpt.old")
    _mk(tmp_path / "ckpt.new", True)
    assert (_ckpt_prepare(tmp_path) / "x").read_text() == "ckpt.new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt"]
    # only an old one survived
    (tmp_path / "ckpt").rename(tmp_path / "ckpt.old")
    assert (_ckpt_prepare(tmp_path) / "x").read_text() == "ckpt.new"
    # nothing complete
    for p in tmp_path.iterdir():
        (p / "progress.json").unlink()
    assert _ckpt_prepare(tmp_path) is None


def _train(tmp_path, cfg, overrides, pause_at=0):
    env = {**os.environ, "GTB_PAUSE_FILE": str(tmp_path / "PAUSE"), "GTB_PAUSE_AT_STEPS": str(pause_at)}
    p = subprocess.run([sys.executable, "scripts/train.py", cfg, *overrides, "--root", str(tmp_path / "res")],
                       env=env, capture_output=True, text=True, check=False)
    return p.returncode, p.stdout + p.stderr


def _eval_set(tmp_path):
    import yaml

    from gtb.envs.variants import load_eval_set

    es = load_eval_set()
    f = tmp_path / "es.yaml"
    f.write_text(yaml.safe_dump({"nominal": es["nominal"], "ood": es["ood"][:1]}))
    return f"train.eval_set={f}"


def _run_dir(tmp_path, method):
    (rd,) = (tmp_path / "res" / method).glob("*/seed0")
    return rd


@pytest.mark.slow
def test_dummy_pause_resume_exact_budget(tmp_path):
    ov = ["train.steps=450", "train.eval_every=100000", "train.eval_workers=2", _eval_set(tmp_path)]
    rc, out = _train(tmp_path, "configs/agent/dummy.yaml", ov, pause_at=150)
    assert rc == EXIT_PAUSED, out
    rd = _run_dir(tmp_path, "dummy")
    assert json.loads((rd / "ckpt" / "progress.json").read_text())["steps"] == 150
    with (rd / "metrics.jsonl").open("a") as f:  # a line logged after the checkpoint, before a crash
        f.write(json.dumps({"steps": 999, "junk": True}) + "\n")
    rc, out = _train(tmp_path, "configs/agent/dummy.yaml", ov)
    assert rc == 0, out
    assert json.loads((rd / "eval.json").read_text())["env_steps"] == 450
    lines = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    assert not any(x.get("junk") for x in lines)
    eps = [x["steps"] for x in lines if "episode_return" in x]
    assert eps == [150, 300, 450]  # every episode logged exactly once
    assert sum(x.get("resumed", False) for x in lines) == 1
    assert not (rd / "ckpt").exists()  # cleaned up when finished


@pytest.mark.slow
def test_mbrl_resume_is_bit_identical(tmp_path):
    import torch

    ov = ["train.steps=900", "train.eval_every=100000", "train.eval_workers=2", "device=cpu", _eval_set(tmp_path),
          "agent.init_steps=300", "agent.fit_steps_first=30", "agent.fit_steps=10", "agent.workers=2",
          "agent.members=2", "agent.hidden=16", "agent.plan.samples=8", "agent.plan.iters=1", "agent.plan.horizon=2"]
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    assert _train(a, "configs/agent/mbrl.yaml", ov)[0] == 0
    rc, out = _train(b, "configs/agent/mbrl.yaml", ov, pause_at=301)  # first safe point after the init data
    assert rc == EXIT_PAUSED, out
    saved = json.loads((_run_dir(b, "mbrl") / "ckpt" / "progress.json").read_text())["steps"]
    assert 301 <= saved < 900  # episodes can end early (TORAX failure), so not always a multiple of 150
    assert _train(b, "configs/agent/mbrl.yaml", ov)[0] == 0
    ra, rb = _run_dir(a, "mbrl"), _run_dir(b, "mbrl")
    ma = torch.load(ra / "model.pt", weights_only=False)["state"]
    mb = torch.load(rb / "model.pt", weights_only=False)["state"]
    assert all(torch.equal(ma[k], mb[k]) for k in ma)
    ea, eb = (json.loads((r / "eval.json").read_text()) for r in (ra, rb))
    assert ea["env_steps"] == eb["env_steps"] >= 900
    assert [e["return"] for e in ea["episodes"]["ood"]] == [e["return"] for e in eb["episodes"]["ood"]]


@pytest.mark.slow
@pytest.mark.parametrize("algo,ov,pause_at", [
    ("sac", ["train.steps=600", "agent.learning_starts=200", "agent.batch_size=32"], 300),
    ("ppo", ["train.steps=900", "agent.n_envs=2", "agent.n_steps=150", "agent.batch_size=100", "agent.n_epochs=1"],
     300),
])
def test_sb3_pause_resume(tmp_path, algo, ov, pause_at):
    from stable_baselines3 import PPO, SAC

    ov = [*ov, "train.eval_every=100000", "train.eval_workers=2", _eval_set(tmp_path)]
    rc, out = _train(tmp_path, f"configs/agent/{algo}.yaml", ov, pause_at=pause_at)
    assert rc == EXIT_PAUSED, out
    rd = _run_dir(tmp_path, algo)
    saved = json.loads((rd / "ckpt" / "progress.json").read_text())["steps"]
    assert saved == json.loads((rd / "ckpt" / "sb3.json").read_text())["num_timesteps"] >= pause_at
    if algo == "sac":
        assert (rd / "ckpt" / "replay.pkl").exists()
    rc, out = _train(tmp_path, f"configs/agent/{algo}.yaml", ov)
    assert rc == 0, out
    budget = int(ov[0].split("=")[1])
    assert json.loads((rd / "eval.json").read_text())["env_steps"] == budget
    model = {"sac": SAC, "ppo": PPO}[algo].load(rd / "model.zip", device="cpu")
    assert model.num_timesteps == budget


@pytest.mark.slow
def test_sweep_pause_and_resume(tmp_path):
    sw = tmp_path / "sw.yaml"
    sw.write_text(json.dumps({
        "seeds": [0, 1], "cpu_slots": 4, "start_gap_s": 0,
        "methods": {"dummy": {"config": "configs/agent/dummy.yaml", "slots": 2, "mem_gb": 1,
                              "overrides": ["train.steps=1500", "train.eval_every=100000", _eval_set(tmp_path)],
                              "resources": ["train.eval_workers=1"]}}}))
    env = {**os.environ, "GTB_PAUSE_FILE": str(tmp_path / "PAUSE")}
    cmd = [sys.executable, "scripts/sweep.py", str(sw), "--root", str(tmp_path / "res")]
    import time

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 300
    def first_episode_logged():
        return any("episode_return" in m.read_text() for m in (tmp_path / "res").glob("dummy/*/seed*/metrics.jsonl"))

    while time.time() < deadline and not first_episode_logged():
        time.sleep(1)
    (tmp_path / "PAUSE").touch()
    out = p.communicate(timeout=300)[0]
    assert p.returncode == EXIT_PAUSED, out
    assert "paused dummy" in out
    (tmp_path / "PAUSE").unlink()
    p2 = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900, check=False)
    assert p2.returncode == 0, p2.stdout + p2.stderr
    assert "(resuming)" in p2.stdout
    for rd in (tmp_path / "res" / "dummy").glob("*/seed*"):
        assert json.loads((rd / "eval.json").read_text())["env_steps"] == 1500
