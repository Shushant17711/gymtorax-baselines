"""Dynamics models over the flat observation, in torch.

Observation layout (gtb.envs.wrapper): [phys (N_PHYS, normalised+clipped) | prev action (ACT_DIM) | t/T].
Models predict only the next phys block; prev action and time advance deterministically.
A model may define a physics `prior_delta`; the network then learns the residual (task 7).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import nn

from gtb.envs.wrapper import ACT_DIM, CLIP, EPISODE_LEN, FEATURE_NAMES, N_PHYS, OBS_DIM

IDX = {k: FEATURE_NAMES.index(k) for k in ("Q_fusion", "H98", "q95", "q_min")}
IDX["T_e0"], IDX["T_i0"] = FEATURE_NAMES.index("T_e[0]"), FEATURE_NAMES.index("T_i[0]")


class EnsembleMLP(nn.Module):
    """E independent MLPs evaluated as batched matmuls; outputs Gaussian mean and bounded log-variance."""

    def __init__(self, in_dim: int, out_dim: int, members: int = 5, hidden: int = 200, layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden] * layers + [2 * out_dim]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for i, o in itertools.pairwise(dims):
            w = torch.empty(members, i, o)
            for e in range(members):
                nn.init.trunc_normal_(w[e], std=1 / (2 * np.sqrt(i)))
            self.W.append(nn.Parameter(w))
            self.b.append(nn.Parameter(torch.zeros(members, 1, o)))
        self.max_logvar = nn.Parameter(torch.full((out_dim,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((out_dim,), -10.0))
        self.members, self.out_dim = members, out_dim

    def forward(self, x: torch.Tensor):  # x: (E, B, in)
        for i, (w, b) in enumerate(zip(self.W, self.b, strict=True)):
            x = torch.baddbmm(b, x, w)
            if i < len(self.W) - 1:
                x = nn.functional.silu(x)
        mu, lv = x.split(self.out_dim, dim=-1)
        lv = self.max_logvar - nn.functional.softplus(self.max_logvar - lv)
        lv = self.min_logvar + nn.functional.softplus(lv - self.min_logvar)
        return mu, lv


class DeltaEnsemble(nn.Module):
    """Probabilistic ensemble predicting phys_{t+1} - phys_t - prior_delta (standardised)."""

    kind = "ensemble"

    def __init__(self, members: int = 5, hidden: int = 200, layers: int = 3, norm: dict | None = None):
        super().__init__()
        self.net = EnsembleMLP(OBS_DIM + ACT_DIM, N_PHYS, members, hidden, layers)
        self.register_buffer("d_mean", torch.zeros(N_PHYS))
        self.register_buffer("d_std", torch.ones(N_PHYS))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))
        self.norm = norm  # {"mean": array, "std": array} in physical units; used by physics priors
        self.kwargs = {"members": members, "hidden": hidden, "layers": layers}

    @property
    def members(self) -> int:
        return self.net.members

    def prior_delta(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*obs.shape[:-1], N_PHYS, device=obs.device, dtype=obs.dtype)

    def _inp(self, obs, act):
        x = torch.cat([obs, act], -1)
        return x if x.dim() == 3 else x.unsqueeze(0).expand(self.members, -1, -1)

    def predict_next_phys(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """obs: (B, OBS) or (E, B, OBS); returns mean next phys (E, B, N_PHYS), clipped like the env."""
        mu, _ = self.net(self._inp(obs, act))
        nxt = obs[..., :N_PHYS] + self.prior_delta(obs, act) + mu * self.d_std + self.d_mean
        return nxt.clamp(-CLIP, CLIP)

    def target(self, obs, act, next_phys):
        return (next_phys - obs[..., :N_PHYS] - self.prior_delta(obs, act) - self.d_mean) / self.d_std

    def nll(self, obs, act, next_phys):
        """obs/act/next_phys: (E, B, .) (each member its own bootstrap batch)."""
        mu, lv = self.net(self._inp(obs, act))
        y = self.target(obs, act, next_phys)
        loss = (((mu - y) ** 2) * torch.exp(-lv) + lv).mean()
        return loss + 0.01 * (self.net.max_logvar.sum() - self.net.min_logvar.sum()) / N_PHYS

    @torch.no_grad()
    def mse(self, obs, act, next_phys) -> float:
        """Mean squared error of the ensemble-mean prediction in normalised feature units."""
        return float(((self.predict_next_phys(obs, act).mean(0) - next_phys) ** 2).mean())

    def fit(self, obs: np.ndarray, act: np.ndarray, next_phys: np.ndarray, steps: int = 2000, batch: int = 256,
            lr: float = 1e-3, wd: float = 1e-5, holdout: float = 0.1, device: str = "cpu", seed: int = 0) -> dict:
        dev = torch.device(device)
        self.to(dev)
        rng = np.random.default_rng(seed)
        n = len(obs)
        perm = rng.permutation(n)
        n_ho = max(1, int(holdout * n))
        ho, tr = perm[:n_ho], perm[n_ho:]
        T = {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in
             {"o": obs, "a": act, "y": next_phys}.items()}
        if not bool(self.fitted):
            with torch.no_grad():
                d = T["y"][tr] - T["o"][tr, :N_PHYS] - self.prior_delta(T["o"][tr], T["a"][tr])
                self.d_mean.copy_(d.mean(0))
                self.d_std.copy_(d.std(0).clamp_min(1e-4))
                self.fitted.fill_(True)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        boot = torch.as_tensor(rng.choice(tr, size=(self.members, len(tr))), device=dev)  # bootstrap per member
        self.train()
        for _ in range(steps):
            j = torch.randint(0, len(tr), (self.members, batch), device=dev)
            idx = torch.gather(boot, 1, j)
            loss = self.nll(T["o"][idx], T["a"][idx], T["y"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.eval()
        hot = torch.as_tensor(ho, device=dev)
        return {"train_nll": float(loss.detach()), "holdout_mse": self.mse(T["o"][hot], T["a"][hot], T["y"][hot]),
                "n_data": n}


def reward_from_phys(phys: torch.Tensor, norm_mean: torch.Tensor, norm_std: torch.Tensor) -> torch.Tensor:
    """The env reward as a function of (normalised) next-state features; mirrors wrapper.reward_components."""
    def x(k):
        i = IDX[k]
        return phys[..., i] * norm_std[i] + norm_mean[i]

    from gtb.envs.wrapper import get_q_cap

    h = ((x("T_e0") > 10) & (x("T_i0") > 10)).to(phys.dtype)
    q, cap = x("Q_fusion"), get_q_cap()
    if cap is not None:
        q = q.clamp(max=cap)
    return (h * q / 10 / 50 + h * x("H98").clamp(max=1.0) / 50
            + x("q_min").clamp(max=1.0) / 150 + (x("q95") / 3).clamp(max=1.0) / 150)


def next_obs(obs: torch.Tensor, next_phys: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    t = obs[..., -1:] + 1.0 / EPISODE_LEN
    return torch.cat([next_phys, act.expand(*next_phys.shape[:-1], ACT_DIM), t], -1)
