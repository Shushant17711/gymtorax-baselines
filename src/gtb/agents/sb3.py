"""PPO and SAC from stable-baselines3 on the flat env, training scenarios drawn per reset.

Vector envs use spawn-started subprocesses (JAX is not fork-safe).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np

from gtb.agents.base import Agent, TrainContext


def _make_callback(ctx: TrainContext, n_envs: int):
    from stable_baselines3.common.callbacks import BaseCallback

    class Tick(BaseCallback):
        def _on_step(self) -> bool:
            rec = None
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    rec = {"episode_return": float(info["episode"]["r"])}
                    ctx.tick(0, rec)
            ctx.tick(n_envs, None)
            return True

    return Tick()


def _make_monitored(seed: int, norm_path: str | None):
    from stable_baselines3.common.monitor import Monitor

    from gtb.envs.train_env import make_train_env

    return Monitor(make_train_env(seed=seed, norm_path=norm_path))


def _env_fns(ctx: TrainContext, n: int, norm_path: str | None):
    """Picklable factories only (no ctx closures: ctx holds the open metrics file)."""
    return [partial(_make_monitored, ctx.seed * 1000 + i, norm_path) for i in range(n)]


class _SB3Agent(Agent):
    algo = ""

    def _vec_env(self, ctx: TrainContext):
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        n = int(self.cfg.agent.get("n_envs", 1))
        fns = _env_fns(ctx, n, self.cfg.env.norm)
        return (SubprocVecEnv(fns, start_method="spawn") if n > 1 else DummyVecEnv(fns)), n

    def _build(self, env, ctx):
        raise NotImplementedError

    def learn(self, ctx: TrainContext) -> None:
        import torch

        torch.set_num_threads(int(self.cfg.agent.get("torch_threads", 4)))
        env, n = self._vec_env(ctx)
        self.model = self._build(env, ctx)
        self.model.learn(total_timesteps=ctx.budget, callback=_make_callback(ctx, n), progress_bar=False)
        env.close()

    def save(self) -> None:
        self.model.save(self.run_dir / "model.zip")

    @classmethod
    def policy_factory(cls, run_dir: str):
        return partial(SB3Policy, cls.algo, str(Path(run_dir) / "model.zip"))


class PPOAgent(_SB3Agent):
    name = algo = "ppo"

    def _build(self, env, ctx):
        from stable_baselines3 import PPO

        a = self.cfg.agent
        return PPO("MlpPolicy", env, learning_rate=a.lr, n_steps=a.n_steps, batch_size=a.batch_size,
                   n_epochs=a.n_epochs, gamma=a.gamma, gae_lambda=a.gae_lambda, clip_range=a.clip_range,
                   ent_coef=a.ent_coef, policy_kwargs={"net_arch": list(a.net_arch)}, seed=ctx.seed,
                   device=ctx.device, verbose=0)


class SACAgent(_SB3Agent):
    name = algo = "sac"

    def _build(self, env, ctx):
        from stable_baselines3 import SAC

        a = self.cfg.agent
        return SAC("MlpPolicy", env, learning_rate=a.lr, buffer_size=a.buffer_size, batch_size=a.batch_size,
                   learning_starts=a.learning_starts, gamma=a.gamma, tau=a.tau, train_freq=a.train_freq,
                   gradient_steps=a.gradient_steps, policy_kwargs={"net_arch": list(a.net_arch)}, seed=ctx.seed,
                   device=ctx.device, verbose=0)


class SB3Policy:
    """Deterministic SB3 policy for evaluation workers (CPU, 1 thread)."""

    def __init__(self, algo: str, path: str):
        import stable_baselines3 as sb3
        import torch

        torch.set_num_threads(1)
        self.model = {"ppo": sb3.PPO, "sac": sb3.SAC}[algo].load(path, device="cpu")

    def reset(self, env):
        pass

    def act(self, obs, env):
        a, _ = self.model.predict(np.asarray(obs, np.float32), deterministic=True)
        return a
