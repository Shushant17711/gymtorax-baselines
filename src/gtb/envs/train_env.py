"""Training env: FlatEnv that draws a fresh training-split scenario at every reset."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from gtb.envs.variants import nominal, sample_variant
from gtb.envs.wrapper import FlatEnv, load_norm


class TrainEnv(gym.Env):
    metadata = {"render_modes": []}  # noqa: RUF012

    def __init__(self, seed: int = 0, split: str = "train", norm_path: str | None = "configs/env/norm.json"):
        self.flat = FlatEnv(norm=load_norm(norm_path) if norm_path else None)
        self.action_space, self.observation_space = self.flat.action_space, self.flat.observation_space
        self.split, self.rng = split, np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        v = nominal() if self.split == "nominal" else sample_variant(self.rng, self.split)
        obs, info = self.flat.reset(options={"variant": v})
        return obs, {**info, "variant": v}

    def step(self, a):
        return self.flat.step(a)

    @property
    def last_state(self):
        return self.flat.last_state

    def close(self):
        self.flat.env.close()


def make_train_env(seed: int = 0, split: str = "train", norm_path: str | None = "configs/env/norm.json"):
    return TrainEnv(seed=seed, split=split, norm_path=norm_path)
