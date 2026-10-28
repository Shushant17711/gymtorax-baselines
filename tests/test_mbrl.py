import json

import numpy as np
import pytest
import torch

from gtb.agents.mbrl import cem
from gtb.agents.models import DeltaEnsemble, reward_from_phys
from gtb.envs.wrapper import ACT_DIM, N_PHYS, OBS_DIM


def test_ensemble_overfits_tiny_dataset():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(256, OBS_DIM)).astype(np.float32)
    act = rng.uniform(-1, 1, (256, ACT_DIM)).astype(np.float32)
    A = rng.normal(size=(ACT_DIM, N_PHYS)) * 0.3
    nxt = (0.9 * obs[:, :N_PHYS] + np.tanh(act @ A)).astype(np.float32)
    m = DeltaEnsemble(members=3, hidden=64, layers=2)
    m.fit(obs, act, nxt, steps=1500, batch=64, lr=3e-3, holdout=0.05)
    t = {k: torch.as_tensor(v) for k, v in {"o": obs, "a": act, "y": nxt}.items()}
    base = float(((t["y"] - t["o"][:, :N_PHYS]) ** 2).mean())
    assert m.mse(t["o"], t["a"], t["y"]) < 0.1 * base  # bootstrap members each miss ~37% of points


def test_cem_solves_quadratic():
    target = torch.linspace(-0.8, 0.8, 3 * 2).reshape(3, 2)
    best = cem(lambda s: -((s - target) ** 2).sum((1, 2)), 3, 2, n=256, iters=8,
               gen=torch.Generator().manual_seed(0))
    assert float(((best - target) ** 2).sum()) < 0.05


@pytest.mark.slow
def test_reward_model_matches_env():
    from gtb.agents.reference import ScriptedPolicy
    from gtb.envs.variants import load_eval_set
    from gtb.envs.wrapper import FlatEnv, load_norm
    from gtb.eval.rollout import run_episode

    norm = load_norm()
    env = FlatEnv(norm=norm)
    nm, ns = (torch.as_tensor(norm[k], dtype=torch.float64) for k in ("mean", "std"))
    for v in (load_eval_set()["nominal"][0], load_eval_set()["ood"][0]):
        r = run_episode(env, ScriptedPolicy(), v, record=True)
        o = torch.as_tensor(r["traj"]["obs"][1:, :N_PHYS], dtype=torch.float64)
        pred = reward_from_phys(o, nm, ns).numpy()
        assert np.abs(pred - r["traj"]["rew"]).max() < 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["mbrl", "pimbrl"])
def test_mbrl_smoke(tmp_path, algo):
    from gtb.config import load_config
    from gtb.train import train

    cfg = load_config(["configs/base.yaml", f"configs/agent/{algo}.yaml"],
                      ["train.steps=600", "train.eval_every=100000", "train.eval_workers=8", "device=cpu",
                       "agent.init_steps=300", "agent.fit_steps_first=50", "agent.fit_steps=20", "agent.workers=2",
                       "agent.members=2", "agent.hidden=32", "agent.plan.samples=16", "agent.plan.iters=2",
                       "agent.plan.horizon=3"])
    rd = train(cfg, root=str(tmp_path))
    lines = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    assert sum("fit" in x for x in lines) >= 2 and any(x.get("source") == "mpc" for x in lines)
    ev = json.loads((rd / "eval.json").read_text())
    assert ev["env_steps"] >= 600 and ev["summary"]["heldout"]["n"] == 15
