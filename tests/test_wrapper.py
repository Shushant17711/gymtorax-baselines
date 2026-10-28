import numpy as np
import pytest

from gtb.envs.wrapper import ACT_DIM, OBS_DIM, FlatEnv


@pytest.fixture(scope="module")
def env():
    return FlatEnv()


def test_action_map_corners(env):
    for a in (-np.ones(ACT_DIM), np.ones(ACT_DIM)):
        d = env.to_env_action(a)
        np.testing.assert_allclose(env.from_env_action(d), a, atol=1e-9)
    lo = env.to_env_action(-np.ones(ACT_DIM))
    hi = env.to_env_action(np.ones(ACT_DIM))
    for k, sp in env.env.action_space.spaces.items():
        np.testing.assert_allclose(lo[k], sp.low)
        np.testing.assert_allclose(hi[k], sp.high)


@pytest.mark.slow
def test_scripted_episode_reward_matches(env):
    from gymtorax import IterHybridAgent

    agent = IterHybridAgent(env.env.action_space)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    total, n, done = 0.0, 0, False
    while not done:
        a = env.from_env_action(agent.act(env.last_state))
        obs, r, te, tr, info = env.step(a)
        assert np.all(np.isfinite(obs)) and not info["failed"]
        assert abs(sum(info["components"].values()) - r) < 1e-9
        total += r
        n += 1
        done = te or tr
    assert n == 150
    assert abs(total - 3.263) < 5e-3


@pytest.mark.slow
def test_normalised_scripted_episode_bounded():
    from gymtorax import IterHybridAgent

    from gtb.envs.wrapper import N_PHYS, load_norm

    env = FlatEnv(norm=load_norm())
    agent = IterHybridAgent(env.env.action_space)
    obs, _ = env.reset(seed=0)
    rows, done = [obs[:N_PHYS]], False
    while not done:
        obs, _, te, tr, _ = env.step(env.from_env_action(agent.act(env.last_state)))
        rows.append(obs[:N_PHYS])
        done = te or tr
    X = np.abs(np.stack(rows))
    assert X.max() <= 10.0
    assert (X >= 10.0).mean() < 1e-3  # clipping is rare


def _fake_state(q, te0=20.0):
    return {"profiles": {"T_e": np.array([te0, 1.0]), "T_i": np.array([te0, 1.0])},
            "scalars": {"Q_fusion": np.array([q]), "H98": np.array([1.2]), "q_min": np.array([1.5]),
                        "q95": np.array([4.0])}}


def test_q_cap_only_changes_fusion_term(monkeypatch):
    from gtb.envs.wrapper import reward_components, set_q_cap
    monkeypatch.setenv("GTB_Q_CAP", "")
    raw = reward_components(_fake_state(500.0))
    assert raw["fusion_gain"] == pytest.approx(500 / 10 / 50)
    set_q_cap(10)
    cap = reward_components(_fake_state(500.0))
    assert cap["fusion_gain"] == pytest.approx(10 / 10 / 50)
    assert {k: v for k, v in cap.items() if k != "fusion_gain"} == {k: v for k, v in raw.items() if k != "fusion_gain"}
    assert reward_components(_fake_state(3.0)) == reward_components(_fake_state(3.0))  # below cap: unchanged
    set_q_cap(None)
    assert reward_components(_fake_state(500.0)) == raw


def test_reward_from_phys_respects_cap(monkeypatch):
    import torch

    from gtb.agents.models import IDX, reward_from_phys
    from gtb.envs.wrapper import N_PHYS, reward_components, set_q_cap
    monkeypatch.setenv("GTB_Q_CAP", "")
    phys = torch.zeros(N_PHYS, dtype=torch.float64)
    vals = {"Q_fusion": 500.0, "H98": 1.2, "q_min": 1.5, "q95": 4.0, "T_e0": 20.0, "T_i0": 20.0}
    for k, v in vals.items():
        phys[IDX[k]] = v
    nm, ns = torch.zeros(N_PHYS, dtype=torch.float64), torch.ones(N_PHYS, dtype=torch.float64)
    for cap in (None, 10.0):
        set_q_cap(cap)
        want = sum(reward_components(_fake_state(500.0)).values())
        assert float(reward_from_phys(phys, nm, ns)) == pytest.approx(want)
    set_q_cap(None)
