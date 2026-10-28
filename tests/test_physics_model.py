import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def episodes():
    """Scripted + noisy-scripted + random episodes on train scenarios, and scripted on held-out scenarios."""
    from functools import partial

    from gtb.agents.mbrl import _noisy_scripted, _random
    from gtb.agents.reference import scripted_policy
    from gtb.envs.variants import load_eval_set, sample_variant
    from gtb.eval.rollout import run_jobs

    rng = np.random.default_rng(0)
    fs = [scripted_policy] * 3 + [partial(_noisy_scripted, 0.2, s) for s in range(3)] + \
         [partial(_random, s) for s in range(3)]
    train = run_jobs([(f, sample_variant(rng, "train"), 0, "configs/env/norm.json", True) for f in fs], workers=12)
    held = run_jobs([(scripted_policy, v, 0, "configs/env/norm.json", True)
                     for v in load_eval_set()["heldout"][:3]], workers=12)
    return train, held


def _arrays(recs):
    from gtb.envs.wrapper import N_PHYS

    o = np.concatenate([r["traj"]["obs"][:-1] for r in recs])
    a = np.concatenate([r["traj"]["act"] for r in recs])
    y = np.concatenate([r["traj"]["obs"][1:, :N_PHYS] for r in recs])
    return o, a, y


def test_action_bounds_match_env():
    from gtb.agents.physics_model import A_HIGH, A_LOW
    from gtb.envs.wrapper import FlatEnv

    env = FlatEnv()
    assert np.allclose(env.low, A_LOW) and np.allclose(env.high, A_HIGH)


@pytest.mark.slow
def test_prior_predicts_stored_energy_change(episodes):
    from gtb.agents.physics_model import I_W, PowerBalanceEnsemble, fit_gain
    from gtb.envs.wrapper import load_norm

    train, held = episodes
    norm = load_norm()
    k = fit_gain(*_arrays(train), norm["mean"], norm["std"])
    assert 0.05 < k < 0.6
    m = PowerBalanceEnsemble(norm=norm)
    m.k.fill_(k)
    o, a, y = _arrays(held)
    with torch.no_grad():
        dp = m.prior_delta(torch.as_tensor(o), torch.as_tensor(a))[:, I_W].numpy()
    dt = y[:, I_W] - o[:, I_W]
    r2 = 1 - ((dt - dp) ** 2).sum() / ((dt - dt.mean()) ** 2).sum()
    print(f"k={k:.3f} heldout R2={r2:.3f}")
    assert r2 > 0.5, r2


@pytest.mark.slow
def test_plugs_into_planner(episodes):
    from gtb.agents.mbrl import MPCPolicy
    from gtb.agents.physics_model import PowerBalanceEnsemble
    from gtb.envs.wrapper import load_norm

    norm = load_norm()
    m = PowerBalanceEnsemble(members=2, hidden=32, layers=2, norm=norm)
    info = m.fit(*_arrays(episodes[0]), steps=50, batch=64)
    assert 0.05 < info["k"] < 0.6
    p = MPCPolicy(m, {"horizon": 3, "samples": 16, "iters": 2, "elite_frac": 0.25, "init_std": 0.5}, norm)
    p.reset(None)
    a = p.act(episodes[1][0]["traj"]["obs"][0], None)
    assert a.shape == (7,) and np.all(np.abs(a) <= 1)
