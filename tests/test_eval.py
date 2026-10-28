import numpy as np
import pytest

from gtb.eval.metrics import iqm, prob_improvement, stratified_bootstrap_ci


def test_iqm_known():
    assert iqm(np.arange(1, 9)) == pytest.approx(4.5)
    assert iqm([0, 0, 1, 1, 1, 1, 100, -100]) == pytest.approx(0.75)  # middle half [0,1,1,1]


def test_bootstrap_ci_contains_point():
    s = np.random.default_rng(0).normal(size=(5, 4))
    p, lo, hi = stratified_bootstrap_ci(s, n_boot=300)
    assert lo <= p <= hi and lo < hi


def test_prob_improvement():
    x, y = np.ones((5, 3)), np.zeros((5, 3))
    assert prob_improvement(x, y) == 1.0 and prob_improvement(y, x) == 0.0 and prob_improvement(x, x) == 0.5


@pytest.mark.slow
def test_random_rollout_reproducible():
    from functools import partial

    from gtb.agents.reference import random_policy
    from gtb.envs.variants import nominal
    from gtb.eval.rollout import evaluate

    recs = evaluate(partial(random_policy, 3), [nominal(), nominal()], workers=2)
    assert recs[0]["return"] == recs[1]["return"] and recs[0]["steps"] == 150
    comps = sum(recs[0][f"ret_{k}"] for k in ("fusion_gain", "h98", "q_min", "q_edge"))
    assert comps == pytest.approx(recs[0]["return"])


def test_classical_ranges_contain_scripted():
    from gtb.agents.reference import PARAM_RANGES, SCRIPTED_PARAMS

    assert all(lo <= SCRIPTED_PARAMS[k] <= hi for k, (lo, hi) in PARAM_RANGES.items())


@pytest.mark.slow
def test_classical_with_scripted_params_reproduces_scripted():
    from functools import partial

    from gtb.agents.reference import classical_policy, scripted_policy
    from gtb.envs.variants import nominal
    from gtb.eval.rollout import evaluate

    c, s = evaluate(partial(classical_policy, None), [nominal()], workers=1) + \
        evaluate(scripted_policy, [nominal()], workers=1)
    assert c["return"] == pytest.approx(s["return"], abs=1e-9)
    fb = evaluate(partial(classical_policy, {"kp": 2.0, "ki": 0.1, "q_target": 1.2}), [nominal()], workers=1)[0]
    assert fb["steps"] == 150 and abs(fb["return"] - s["return"]) > 1e-4  # feedback path is live


def test_iqm_diff_ci():
    from gtb.eval.metrics import iqm_diff_ci

    rng = np.random.default_rng(1)
    base = rng.normal(size=(1, 20))
    x = base + 1.0 + 0.1 * rng.normal(size=(5, 20))
    y = base + 0.1 * rng.normal(size=(5, 20))
    d, lo, hi = iqm_diff_ci(x, y, n_boot=500)
    assert lo <= d <= hi and lo > 0.5
    d, lo, hi = iqm_diff_ci(y, y + 0.0, n_boot=500)
    assert d == 0 and lo < 0 < hi
    with pytest.raises(ValueError):
        iqm_diff_ci(x, y[:, :5])
