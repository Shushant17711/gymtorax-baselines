import numpy as np
import pytest
from gymtorax.envs.iter_hybrid_env import CONFIG

from gtb.envs.variants import NOMINAL, TRAIN_RANGE, apply_variant, load_eval_set, nominal, sample_variant


def test_nominal_is_identity():
    assert apply_variant(CONFIG, nominal())["profile_conditions"] == CONFIG["profile_conditions"]
    assert apply_variant(CONFIG, nominal())["pedestal"] == CONFIG["pedestal"]
    assert apply_variant(CONFIG, nominal())["plasma_composition"] == CONFIG["plasma_composition"]


def test_apply_does_not_mutate_and_scales():
    before = repr(CONFIG)
    c = apply_variant(CONFIG, {"T_init": 1.1, "T_ped": 0.9, "nbar": 0.8, "Z_eff": 1.8, "coll": 1.2})
    assert repr(CONFIG) == before
    assert c["profile_conditions"]["T_e"][0.0][0.0] == pytest.approx(6.6)
    assert c["pedestal"]["T_i_ped"][105] == pytest.approx(2.7)
    assert c["transport"]["collisionality_multiplier"] == 1.2
    with pytest.raises(KeyError):
        apply_variant(CONFIG, {"bogus": 1.0})


def test_sampling_ranges():
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = sample_variant(rng, "train")
        assert all(TRAIN_RANGE[k][0] <= v[k] <= TRAIN_RANGE[k][1] for k in v)
        o = sample_variant(rng, "ood")
        assert any(not (TRAIN_RANGE[k][0] <= o[k] <= TRAIN_RANGE[k][1]) for k in o)
    assert sample_variant(np.random.default_rng(5)) == sample_variant(np.random.default_rng(5))


def test_eval_set_frozen():
    s = load_eval_set()
    assert [len(s[k]) for k in ("nominal", "heldout", "ood")] == [1, 15, 5]
    assert s["nominal"][0] == NOMINAL


@pytest.mark.slow
def test_variant_changes_trajectory_and_nominal_reproduces():
    from gymtorax import IterHybridAgent

    from gtb.envs.wrapper import FlatEnv

    env = FlatEnv()

    def run(v):
        env.reset(options={"variant": v})
        ag, R, done = IterHybridAgent(env.env.action_space), 0.0, False
        while not done:
            _, r, te, tr, _ = env.step(env.from_env_action(ag.act(env.last_state)))
            R, done = R + r, te or tr
        return R

    v = load_eval_set()["heldout"][0]
    r_nom, r_var, r_nom2 = run(nominal()), run(v), run(nominal())
    assert abs(r_nom - 3.263) < 5e-3 and r_nom == r_nom2
    assert abs(r_var - r_nom) > 1e-3
