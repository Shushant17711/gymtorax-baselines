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
