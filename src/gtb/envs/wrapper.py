"""Flat-vector wrapper around gymtorax/IterHybrid-v0.

Observation: selected scalars + subsampled radial profiles + previous action + t/T.
Action: a in [-1, 1]^7, affinely mapped to the env's Dict(ECRH(3), Ip(1), NBI(3)) Box bounds.
info: per-step reward components (re-implemented from IterHybridEnv._compute_reward), constraint flags, failed.
"""
from __future__ import annotations

import os
from typing import ClassVar

import gymnasium as gym
import gymtorax  # noqa: F401  (registers envs)
import numpy as np

SCALARS = ("Q_fusion", "H98", "q95", "q_min", "W_thermal_total", "P_fusion", "P_aux_total", "tau_E",
           "n_e_line_avg", "fgw_n_e_line_avg", "li3", "T_e_volume_avg", "T_i_volume_avg", "dW_thermal_dt")
PROFILES = ("T_e", "T_i", "n_e", "q", "j_total")
N_RADIAL = 8
ACTION_KEYS = (("ECRH", 3), ("Ip", 1), ("NBI", 3))
ACT_DIM = sum(n for _, n in ACTION_KEYS)
EPISODE_LEN = 150
CLIP = 10.0  # normalised features are clipped; tau_E is heavy-tailed early in the shot


def _scalar(state: dict, k: str) -> float:
    return float(np.asarray(state["scalars"][k]).reshape(-1)[0])


def set_q_cap(q_cap: float | None) -> None:
    """Cap Q in the fusion-gain reward term (None = original uncapped reward, PREREGISTRATION amendment 1).

    Stored in the environment so spawned worker processes (rollouts, MPC planning) inherit it."""
    os.environ["GTB_Q_CAP"] = "" if q_cap is None else str(float(q_cap))


def get_q_cap() -> float | None:
    v = os.environ.get("GTB_Q_CAP", "")
    return float(v) if v else None


def reward_components(state: dict) -> dict[str, float]:
    """Exact re-implementation of IterHybridEnv._compute_reward, as a function of next_state only.

    With a Q cap set (`set_q_cap`), the fusion-gain term uses min(Q, cap)."""
    te0 = float(np.asarray(state["profiles"]["T_e"])[0])
    ti0 = float(np.asarray(state["profiles"]["T_i"])[0])
    h_mode = te0 > 10 and ti0 > 10
    q = _scalar(state, "Q_fusion")
    cap = get_q_cap()
    q_fus = (q if cap is None else min(q, cap)) / 10
    h98 = _scalar(state, "H98")
    q_min = _scalar(state, "q_min")
    q95 = _scalar(state, "q95")
    return {
        "fusion_gain": (q_fus if h_mode else 0.0) / 50,
        "h98": (min(h98, 1.0) if h_mode else 0.0) / 50,
        "q_min": min(q_min, 1.0) / 150,
        "q_edge": min(q95 / 3, 1.0) / 150,
    }


def featurize(state: dict) -> np.ndarray:
    """State dict -> physics feature vector (no action / time features)."""
    feats = [_scalar(state, k) for k in SCALARS]
    for k in PROFILES:
        p = np.asarray(state["profiles"][k], dtype=np.float64).reshape(-1)
        idx = np.linspace(0, len(p) - 1, N_RADIAL).round().astype(int)
        feats.extend(p[idx].tolist())
    return np.asarray(feats, dtype=np.float64)


FEATURE_NAMES = list(SCALARS) + [f"{k}[{i}]" for k in PROFILES for i in range(N_RADIAL)]
N_PHYS = len(FEATURE_NAMES)
OBS_DIM = N_PHYS + ACT_DIM + 1


class FlatEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, variant: dict | None = None, norm: dict | None = None, env_kwargs=None):
        self.env_kwargs = dict(env_kwargs or {})
        self.variant = None
        self._build(variant)
        space = self.env.action_space
        self.low = np.concatenate([np.asarray(space[k].low, np.float64).reshape(-1) for k, _ in ACTION_KEYS])
        self.high = np.concatenate([np.asarray(space[k].high, np.float64).reshape(-1) for k, _ in ACTION_KEYS])
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.norm = norm
        self._t = 0
        self._prev_a = np.zeros(ACT_DIM)
        self.last_state: dict | None = None

    def _build(self, variant: dict | None) -> None:
        from gtb.envs.variants import NOMINAL, VariantIterHybridEnv

        v = {**NOMINAL, **(variant or {})}
        if v == self.variant:
            return
        if getattr(self, "env", None) is not None:
            self.env.close()
        self.env = gym.wrappers.OrderEnforcing(VariantIterHybridEnv(variant=v, **self.env_kwargs))
        self.inner = self.env.unwrapped
        self.variant = v

    # --- action mapping -------------------------------------------------
    def to_env_action(self, a: np.ndarray) -> dict:
        a = np.clip(np.asarray(a, np.float64).reshape(ACT_DIM), -1, 1)
        x = self.low + (a + 1) / 2 * (self.high - self.low)
        out, i = {}, 0
        for k, n in ACTION_KEYS:
            out[k] = x[i:i + n]
            i += n
        return out

    def from_env_action(self, d: dict) -> np.ndarray:
        x = np.concatenate([np.asarray(d[k], np.float64).reshape(-1) for k, _ in ACTION_KEYS])
        return np.clip(2 * (x - self.low) / (self.high - self.low) - 1, -1, 1)

    # --- obs ------------------------------------------------------------
    def _obs(self, state: dict) -> np.ndarray:
        f = featurize(state)
        if self.norm is not None:
            f = np.clip((f - self.norm["mean"]) / self.norm["std"], -CLIP, CLIP)
        return np.concatenate([f, self._prev_a, [self._t / EPISODE_LEN]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        """options={"variant": {...}} switches scenario (rebuilds the TORAX app if it changed)."""
        options = dict(options or {})
        if "variant" in options:
            self._build(options.pop("variant"))
        state, info = self.env.reset(seed=seed, options=options or None)
        self._t, self._prev_a, self.last_state = 0, np.zeros(ACT_DIM), state
        return self._obs(state), dict(info or {})

    def step(self, a):
        a = np.clip(np.asarray(a, np.float64).reshape(ACT_DIM), -1, 1)
        try:
            state, r, term, trunc, info = self.env.step(self.to_env_action(a))
            info = dict(info or {})
            f = featurize(state)
            failed = not np.all(np.isfinite(f))
        except Exception as e:  # noqa: BLE001  TORAX solver failure
            state, r, term, trunc, info, failed = self.last_state, 0.0, True, False, {"error": repr(e)}, True
        self._t += 1
        self._prev_a = a
        self.last_state = state
        comps = reward_components(state)
        if get_q_cap() is not None and not failed:
            r = sum(comps.values())  # capped reward replaces the env's own
        info.update(components=comps, failed=failed,
                    viol_q_min=_scalar(state, "q_min") < 1.0, viol_q95=_scalar(state, "q95") < 3.0)
        obs = self._obs(state)
        if failed:
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            term = True
        return obs, float(r), bool(term), bool(trunc), info


def load_norm(path: str = "configs/env/norm.json") -> dict:
    import json
    from pathlib import Path

    d = json.loads(Path(path).read_text())
    if d["features"] != FEATURE_NAMES:
        raise ValueError("norm.json feature list does not match wrapper FEATURE_NAMES")
    return {"mean": np.asarray(d["mean"]), "std": np.asarray(d["std"])}
