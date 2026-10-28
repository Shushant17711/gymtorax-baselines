"""Reference policies: random, scripted (bundled IterHybridAgent), constant, and PID on the Ip actuator."""
from __future__ import annotations

import numpy as np


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def reset(self, env):
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs, env):
        return self.rng.uniform(-1, 1, env.action_space.shape)


class ScriptedPolicy:
    """The ITER-hybrid feed-forward schedule shipped with gymtorax."""

    def reset(self, env):
        from gymtorax import IterHybridAgent

        self.agent = IterHybridAgent(env.env.action_space)

    def act(self, obs, env):
        return env.from_env_action(self.agent.act(env.last_state))


def random_policy(seed: int = 0):
    return RandomPolicy(seed)


def scripted_policy():
    return ScriptedPolicy()


# ---------------------------------------------------------------------------------------------
# Classical controller: feed-forward schedule + PI feedback of Ip on q_min.
# Physical units (A, W, normalised rho). One action step = 1 s of plasma time.
SCRIPTED_PARAMS = {
    "Ip0": 3.0e6, "Ip_flat": 12.5e6, "t_ramp": 100.0,   # Ip(t) = Ip0 + (Ip_flat-Ip0)*min(1,(t+1)/t_ramp)
    "t_heat": 99.0, "P_nbi": 33e6, "P_ecrh": 20e6,       # heating switched on at t >= t_heat
    "ecrh_loc": 0.35,                                    # ECRH deposition (rho)
    "q_target": 1.0, "kp": 0.0, "ki": 0.0,               # Ip += kp*e + ki*sum(e) [MA per unit q], e = q_min - q_target
}
PARAM_RANGES = {
    "Ip0": (2.0e6, 6.0e6), "Ip_flat": (8.0e6, 15.0e6), "t_ramp": (40.0, 140.0),
    "t_heat": (0.0, 140.0), "P_nbi": (0.0, 33e6), "P_ecrh": (0.0, 20e6), "ecrh_loc": (0.0, 0.7),
    "q_target": (0.8, 1.5), "kp": (0.0, 5.0), "ki": (0.0, 0.5),
}


class ClassicalPolicy:
    """Gain-scheduled feed-forward + PI q_min controller acting on Ip (actuator limits handled by the env)."""

    def __init__(self, params: dict | None = None):
        self.p = {**SCRIPTED_PARAMS, **(params or {})}

    def reset(self, env):
        self.t, self.integral = 0, 0.0

    def act(self, obs, env):
        p, t = self.p, self.t
        ip = p["Ip0"] + (p["Ip_flat"] - p["Ip0"]) * min(1.0, (t + 1) / p["t_ramp"])
        if p["kp"] or p["ki"]:
            q_min = float(np.asarray(env.last_state["scalars"]["q_min"]).reshape(-1)[0])
            e = q_min - p["q_target"]
            self.integral = float(np.clip(self.integral + e, -20, 20))  # anti-windup
            ip += 1e6 * (p["kp"] * e + p["ki"] * self.integral)
        heat = t >= p["t_heat"]
        d = {"Ip": [ip],
             "NBI": [p["P_nbi"] if heat else 0.0, 0.25, 0.25],
             "ECRH": [p["P_ecrh"] if heat else 0.0, p["ecrh_loc"], 0.05]}
        self.t += 1
        return env.from_env_action(d)


def classical_policy(params: dict | None = None):
    return ClassicalPolicy(params)
