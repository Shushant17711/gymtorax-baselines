"""PPO and SAC from stable-baselines3 on the flat env, training scenarios drawn per reset.

Vector envs use spawn-started subprocesses (JAX is not fork-safe).
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np

from gtb.agents.base import Agent, TrainContext


def _make_callback(ctx: TrainContext, n_envs: int, save, on_policy: bool):
    """Counts env steps, logs episode returns, and offers checkpoint/pause points.

    Safe points are rollout starts (the previous rollout is fully trained on): every rollout for PPO
    (one 150-step episode per env), and for SAC only when the last step ended an episode."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Tick(BaseCallback):
        at_episode_end = True

        def _on_rollout_start(self) -> None:
            if on_policy or self.at_episode_end:
                ctx.boundary(save)

        def _on_step(self) -> bool:
            rec = None
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    rec = {"episode_return": float(info["episode"]["r"])}
                    ctx.tick(0, rec)
            ctx.tick(n_envs, None)
            dones = self.locals.get("dones")
            self.at_episode_end = dones is not None and bool(np.all(dones))
            return True

    return Tick()


def _make_monitored(seed: int, norm_path: str | None):
    from stable_baselines3.common.monitor import Monitor

    from gtb.envs.train_env import make_train_env

    return Monitor(make_train_env(seed=seed, norm_path=norm_path))


def _env_seed(seed: int, i: int, done: int) -> int:
    """Fresh runs keep the original seeding; a resumed run draws new scenarios (not a replay of the first ones)."""
    return seed * 1000 + i if done == 0 else int(np.random.SeedSequence([seed, i, done]).generate_state(1)[0])


def _env_fns(ctx: TrainContext, n: int, norm_path: str | None, done: int = 0):
    """Picklable factories only (no ctx closures: ctx holds the open metrics file)."""
    return [partial(_make_monitored, _env_seed(ctx.seed, i, done), norm_path) for i in range(n)]


class _SB3Agent(Agent):
    algo = ""

    def _vec_env(self, ctx: TrainContext, done: int = 0):
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        n = int(self.cfg.agent.get("n_envs", 1))
        fns = _env_fns(ctx, n, self.cfg.env.norm, done)
        return (SubprocVecEnv(fns, start_method="spawn") if n > 1 else DummyVecEnv(fns)), n

    def _build(self, env, ctx):
        raise NotImplementedError

    def _algo_cls(self):
        import stable_baselines3 as sb3

        return {"ppo": sb3.PPO, "sac": sb3.SAC}[self.algo]

    def _save_ckpt(self, d: Path) -> None:
        self.model.save(d / "model.zip")
        if hasattr(self.model, "replay_buffer"):  # off-policy: the buffer is part of the state
            self.model.save_replay_buffer(d / "replay.pkl")
        (d / "sb3.json").write_text(json.dumps({"num_timesteps": int(self.model.num_timesteps)}))

    def learn(self, ctx: TrainContext) -> None:
        import torch

        torch.set_num_threads(int(self.cfg.agent.get("torch_threads", 4)))
        done = json.loads((ctx.resume / "sb3.json").read_text())["num_timesteps"] if ctx.resume else 0
        env, n = self._vec_env(ctx, done)
        try:
            if ctx.resume:
                self.model = self._algo_cls().load(ctx.resume / "model.zip", env=env, device=ctx.device)
                if (ctx.resume / "replay.pkl").exists():
                    self.model.load_replay_buffer(ctx.resume / "replay.pkl")
                assert self.model.num_timesteps == done
            else:
                self.model = self._build(env, ctx)
            cb = _make_callback(ctx, n, self._save_ckpt, on_policy=self.algo == "ppo")
            # reset_num_timesteps=False: SB3 adds the steps already done, so the total stays ctx.budget
            self.model.learn(total_timesteps=ctx.budget - done, callback=cb, progress_bar=False,
                             reset_num_timesteps=not ctx.resume)
        finally:
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
