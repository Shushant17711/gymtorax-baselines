"""Model-based RL: learned dynamics + CEM model-predictive control (PETS-style, mean-propagated ensemble).

learn(): init data from random + noisy-scripted episodes -> fit -> repeat {collect MPC episodes in parallel
workers (CPU planning), refit} until the env-step budget is spent. Training of the model runs on ctx.device.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from gtb.agents.base import Agent, TrainContext
from gtb.agents.models import DeltaEnsemble, next_obs, reward_from_phys
from gtb.envs.wrapper import ACT_DIM, N_PHYS, load_norm


def cem(objective, horizon: int, act_dim: int, n: int = 256, iters: int = 4, elite_frac: float = 0.1,
        init_mean: torch.Tensor | None = None, init_std: float = 0.5, alpha: float = 0.1,
        gen: torch.Generator | None = None) -> torch.Tensor:
    """Maximise objective(actions (n, H, A)) -> (n,) over [-1, 1]^(H x A). Returns the final mean (H, A)."""
    mean = torch.zeros(horizon, act_dim) if init_mean is None else init_mean.clone()
    std = torch.full((horizon, act_dim), init_std)
    k = max(2, int(elite_frac * n))
    best, best_v = mean, -float("inf")
    for _ in range(iters):
        s = (mean + std * torch.randn(n, horizon, act_dim, generator=gen)).clamp(-1, 1)
        s[0] = mean  # always score the current mean
        v = objective(s)
        top = v.topk(k).indices
        if float(v[top[0]]) > best_v:
            best_v, best = float(v[top[0]]), s[top[0]].clone()
        e = s[top]
        mean = alpha * mean + (1 - alpha) * e.mean(0)
        std = alpha * std + (1 - alpha) * e.std(0)
    return best


def model_objective(model: DeltaEnsemble, obs: np.ndarray, nm: torch.Tensor, ns: torch.Tensor):
    o0 = torch.as_tensor(obs, dtype=torch.float32)

    @torch.no_grad()
    def f(acts: torch.Tensor) -> torch.Tensor:  # acts (n, H, A)
        n, H, _ = acts.shape
        o = o0.expand(model.members, n, -1)
        total = torch.zeros(model.members, n)
        for h in range(H):
            a = acts[:, h].unsqueeze(0).expand(model.members, -1, -1)
            nxt = model.predict_next_phys(o, a)
            live = (o[..., -1] < 1.0 - 1e-6).float()  # no reward past the end of the shot
            total += live * reward_from_phys(nxt, nm, ns)
            o = next_obs(o, nxt, a)
        return total.mean(0)

    return f


class MPCPolicy:
    def __init__(self, model: DeltaEnsemble, plan: dict, norm: dict, seed: int = 0, noise: float = 0.0):
        self.model, self.plan, self.seed, self.noise = model.cpu().eval(), dict(plan), seed, noise
        self.nm = torch.as_tensor(norm["mean"], dtype=torch.float32)
        self.ns = torch.as_tensor(norm["std"], dtype=torch.float32)

    def reset(self, env):
        torch.set_num_threads(1)
        self.gen = torch.Generator().manual_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.prev = None

    def act(self, obs, env):
        p = self.plan
        init = None if self.prev is None else torch.cat([self.prev[1:], self.prev[-1:]])
        plan = cem(model_objective(self.model, obs, self.nm, self.ns), p["horizon"], ACT_DIM, n=p["samples"],
                   iters=p["iters"], elite_frac=p["elite_frac"], init_mean=init, init_std=p["init_std"],
                   gen=self.gen)
        self.prev = plan
        a = plan[0].numpy().astype(np.float64)
        if self.noise:
            a = np.clip(a + self.noise * self.rng.normal(size=ACT_DIM), -1, 1)
        return a


class NoisyScripted:
    def __init__(self, sigma: float, seed: int):
        from gtb.agents.reference import ScriptedPolicy

        self.base, self.sigma, self.seed = ScriptedPolicy(), sigma, seed

    def reset(self, env):
        self.base.reset(env)
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs, env):
        return np.clip(self.base.act(obs, env) + self.sigma * self.rng.normal(size=ACT_DIM), -1, 1)


# --- picklable factories for worker processes -------------------------------------------------------------
def _random(seed):
    from gtb.agents.reference import RandomPolicy

    return RandomPolicy(seed)


def _noisy_scripted(sigma, seed):
    return NoisyScripted(sigma, seed)


def _mpc_from_blob(model_kind, model_kwargs, state, plan, norm_path, seed, noise):
    norm = load_norm(norm_path)
    m = MODEL_KINDS[model_kind](**model_kwargs, norm=norm)
    m.load_state_dict(state)
    return MPCPolicy(m, plan, norm, seed=seed, noise=noise)


def _load_mpc(run_dir: str, seed: int = 0):
    import yaml

    rd = Path(run_dir)
    cfg = yaml.safe_load((rd / "config.yaml").read_text())
    blob = torch.load(rd / "model.pt", map_location="cpu", weights_only=False)
    return _mpc_from_blob(blob["kind"], blob["kwargs"], blob["state"], cfg["agent"]["plan"], cfg["env"]["norm"],
                          seed, 0.0)


def _physics_model(**kw):
    from gtb.agents.physics_model import PowerBalanceEnsemble

    return PowerBalanceEnsemble(**kw)


MODEL_KINDS = {"ensemble": DeltaEnsemble, "power_balance": _physics_model}


class MBRLAgent(Agent):
    name = "mbrl"
    model_kind = "ensemble"

    def _new_model(self):
        a = self.cfg.agent
        return MODEL_KINDS[self.model_kind](members=a.members, hidden=a.hidden, layers=a.layers,
                                            norm=load_norm(self.cfg.env.norm))

    def learn(self, ctx: TrainContext) -> None:
        from gtb.envs.variants import sample_variant
        from gtb.eval.rollout import make_pool, run_jobs

        a, norm_path = self.cfg.agent, self.cfg.env.norm
        rng = np.random.default_rng(ctx.seed)
        self.model = self._new_model()
        data = {"obs": [], "act": [], "nxt": [], "src": []}
        steps, n_ep = 0, 0

        def add(recs, src):
            nonlocal steps
            for r in recs:
                t = r["traj"]
                o, u = t["obs"], t["act"]
                k = len(u) - (1 if r["failed"] else 0)  # last transition of a failed shot is garbage
                data["obs"].append(o[:k])
                data["act"].append(u[:k])
                data["nxt"].append(o[1:k + 1, :N_PHYS])
                data["src"].append(np.full(k, src))
                steps += r["steps"]
                ctx.tick(r["steps"], {"episode_return": r["return"], "source": src, "failed": r["failed"]})

        def fit(fit_steps):
            cat = {k: np.concatenate(v) for k, v in data.items()}
            info = self.model.fit(cat["obs"], cat["act"], cat["nxt"], steps=fit_steps, batch=a.batch, lr=a.lr,
                                  device=ctx.device, seed=int(rng.integers(1 << 31)))
            self.model.cpu()
            ctx.tick(0, {"fit": info})

        def jobs(factories):
            return [(f, sample_variant(rng, "train"), 0, norm_path, True) for f in factories]

        with make_pool(int(a.workers)) as pool:
            n_init = max(1, int(a.init_steps) // 150)
            fs = [partial(_random, int(rng.integers(1 << 31))) if i % 2 == 0 else
                  partial(_noisy_scripted, float(a.scripted_noise), int(rng.integers(1 << 31)))
                  for i in range(n_init)]
            add(run_jobs(jobs(fs), pool=pool), "init")
            fit(int(a.fit_steps_first))
            while steps < ctx.budget:
                n = min(int(a.workers), max(1, (ctx.budget - steps) // 150))
                blob = (self.model_kind, self.model.kwargs, self.model.state_dict(), self.cfg.agent.plan.to_dict(),
                        norm_path)
                fs = [partial(_mpc_from_blob, *blob, int(rng.integers(1 << 31)), float(a.explore_noise))
                      for _ in range(n)]
                add(run_jobs(jobs(fs), pool=pool), "mpc")
                n_ep += n
                fit(int(a.fit_steps))
        self.data = data

    def save(self) -> None:
        torch.save({"kind": self.model_kind, "kwargs": self.model.kwargs, "state": self.model.cpu().state_dict()},
                   self.run_dir / "model.pt")
        if getattr(self, "data", None):
            np.savez_compressed(self.run_dir / "data.npz", **{k: np.concatenate(v) for k, v in self.data.items()})
            (self.run_dir / "data.json").write_text(json.dumps({"n": int(sum(map(len, self.data["act"])))}))

    @staticmethod
    def policy_factory(run_dir: str):
        return partial(_load_mpc, run_dir)


class PIMBRLAgent(MBRLAgent):
    name = "pimbrl"
    model_kind = "power_balance"
