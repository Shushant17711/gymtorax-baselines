"""Physics-informed dynamics: a 0-D power-balance prior for the stored energy, learned residual for the rest.

Power balance dW/dt = P_heat - P_loss, with the loss linearised around the current state as P_loss ~ W / tau_E:

    dW_{t+1} = dW_t * exp(-1 / tau_E,t) + k * (P_aux,cmd - P_aux,t)

i.e. last step's energy change decays on the confinement time, and a step in commanded auxiliary heating enters
with an effective gain k (< 1: TORAX's profiles take several seconds to respond). k is the prior's only constant
and is fitted by least squares on the agent's own initial data at the first fit. dW_t is TORAX's dW_thermal_dt
(the previous step's change). Only the W_thermal feature gets a prior; the ensemble learns its residual and the
full delta of every other feature.
"""
from __future__ import annotations

import numpy as np
import torch

from gtb.agents.models import DeltaEnsemble
from gtb.envs.wrapper import FEATURE_NAMES, N_PHYS

# FlatEnv action layout ECRH(3) | Ip(1) | NBI(3); Box bounds of gymtorax/IterHybrid-v0 (checked by a test)
A_LOW = (0.0, 0.0, 0.01, 1e3, 0.0, 0.0, 0.01)
A_HIGH = (20e6, 1.0, 1.0, 15e6, 33e6, 1.0, 1.0)
I_ECRH, I_NBI = 0, 4
I_W, I_TAU, I_PAUX, I_DW = (FEATURE_NAMES.index(k) for k in
                            ("W_thermal_total", "tau_E", "P_aux_total", "dW_thermal_dt"))


def commanded_aux_power(act: torch.Tensor) -> torch.Tensor:
    lo = torch.as_tensor(A_LOW, dtype=act.dtype, device=act.device)
    hi = torch.as_tensor(A_HIGH, dtype=act.dtype, device=act.device)
    u = lo + (act.clamp(-1, 1) + 1) / 2 * (hi - lo)
    return u[..., I_ECRH] + u[..., I_NBI]


def prior_terms(obs: torch.Tensor, act: torch.Tensor, nm: torch.Tensor, ns: torch.Tensor):
    """(decay term, heating step) in J and W. Prior dW = decay + k * step."""
    def phys(i):
        return obs[..., i] * ns[i] + nm[i]

    tau = phys(I_TAU).clamp(0.05, 100.0)
    decay = phys(I_DW) * torch.exp(-1.0 / tau)
    step = commanded_aux_power(act) - phys(I_PAUX).clamp_min(0.0)
    return decay, step


def fit_gain(obs: np.ndarray, act: np.ndarray, next_phys: np.ndarray, nm, ns) -> float:
    o, a = torch.as_tensor(obs, dtype=torch.float64), torch.as_tensor(act, dtype=torch.float64)
    nm, ns = torch.as_tensor(nm, dtype=torch.float64), torch.as_tensor(ns, dtype=torch.float64)
    decay, step = prior_terms(o, a, nm, ns)
    dw = (torch.as_tensor(next_phys[:, I_W], dtype=torch.float64) - o[:, I_W]) * ns[I_W]
    k = float(((dw - decay) * step).sum() / (step**2).sum().clamp_min(1e-12))
    return float(np.clip(k, 0.0, 1.0))


class PowerBalanceEnsemble(DeltaEnsemble):
    kind = "power_balance"

    def __init__(self, members: int = 5, hidden: int = 200, layers: int = 3, norm: dict | None = None):
        super().__init__(members, hidden, layers, norm)
        if norm is None:
            raise ValueError("PowerBalanceEnsemble needs the feature normalisation (physical units)")
        self.register_buffer("nm", torch.as_tensor(norm["mean"], dtype=torch.float32))
        self.register_buffer("ns", torch.as_tensor(norm["std"], dtype=torch.float32))
        self.register_buffer("k", torch.zeros(()))

    def prior_delta(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        nm, ns = self.nm.to(obs.dtype), self.ns.to(obs.dtype)
        decay, step = prior_terms(obs, act, nm, ns)
        d = torch.zeros(*obs.shape[:-1], N_PHYS, device=obs.device, dtype=obs.dtype)
        d[..., I_W] = (decay + self.k.to(obs.dtype) * step) / ns[I_W]
        return d

    def fit(self, obs, act, next_phys, **kw) -> dict:
        if not bool(self.fitted):
            self.k.fill_(fit_gain(obs, act, next_phys, self.nm.cpu().numpy(), self.ns.cpu().numpy()))
        return {**super().fit(obs, act, next_phys, **kw), "k": float(self.k)}
