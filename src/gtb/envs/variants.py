"""Perturbed ITER-hybrid scenarios.

A variant is a dict of 5 scalars; `nominal()` reproduces gymtorax/IterHybrid-v0 exactly.
  T_init   multiplies the initial T_e and T_i profiles            (train range [0.8, 1.2])
  nbar     line-averaged density, Greenwald fraction (nominal 0.85; train [0.75, 0.95])
  T_ped    multiplies the H-mode pedestal temperatures            (train [0.85, 1.15])
  Z_eff    effective charge (impurity content; nominal 2.0)     (train [1.7, 2.3])
  coll     QLKNN collisionality multiplier (transport proxy)     (train [0.8, 1.25], log-uniform)
OOD draws come from the band between the train range and a range twice as wide.
"""
from __future__ import annotations

import copy

import numpy as np
from gymtorax.envs.iter_hybrid_env import CONFIG, IterHybridEnv

NOMINAL = {"T_init": 1.0, "nbar": 0.85, "T_ped": 1.0, "Z_eff": 2.0, "coll": 1.0}
TRAIN_RANGE = {"T_init": (0.8, 1.2), "nbar": (0.75, 0.95), "T_ped": (0.85, 1.15),
               "Z_eff": (1.7, 2.3), "coll": (0.8, 1.25)}
LOG_UNIFORM = {"coll"}


def nominal() -> dict[str, float]:
    return dict(NOMINAL)


def _scale_nested(d: dict, s: float) -> dict:
    return {k: (_scale_nested(v, s) if isinstance(v, dict) else v * s) for k, v in d.items()}


def apply_variant(config: dict, v: dict[str, float]) -> dict:
    """Return a deep copy of a TORAX config dict with the variant applied."""
    unknown = set(v) - set(NOMINAL)
    if unknown:
        raise KeyError(f"unknown variant keys {unknown}")
    v = {**NOMINAL, **v}
    c = copy.deepcopy(config)
    pc = c["profile_conditions"]
    for k in ("T_e", "T_i"):
        pc[k] = _scale_nested(pc[k], v["T_init"])
    pc["nbar"] = v["nbar"]
    ped = c["pedestal"]
    for k in ("T_e_ped", "T_i_ped"):
        ped[k] = {t: x * v["T_ped"] for t, x in ped[k].items()}
    c["plasma_composition"]["Z_eff"] = {0.0: {0.0: v["Z_eff"], 1.0: v["Z_eff"]}}
    c["transport"]["collisionality_multiplier"] = v["coll"]
    return c


def _draw(rng: np.random.Generator, k: str, lo: float, hi: float) -> float:
    if k in LOG_UNIFORM:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def sample_variant(rng: np.random.Generator, split: str = "train") -> dict[str, float]:
    """split='train'/'heldout' -> inside TRAIN_RANGE; 'ood' -> one or more params in the outer band."""
    if split in ("train", "heldout"):
        return {k: _draw(rng, k, lo, hi) for k, (lo, hi) in TRAIN_RANGE.items()}
    if split != "ood":
        raise ValueError(split)
    v = sample_variant(rng, "train")
    keys = list(TRAIN_RANGE)
    for k in rng.choice(keys, size=rng.integers(1, 3), replace=False):
        lo, hi = TRAIN_RANGE[k]
        c = NOMINAL[k]
        if k in LOG_UNIFORM:
            side = rng.choice([-1, 1])
            span = np.log(hi / c) if side > 0 else np.log(c / lo)
            v[k] = float(c * np.exp(side * rng.uniform(span, 2 * span)))
        else:
            side = rng.choice([-1, 1])
            span = (hi - c) if side > 0 else (c - lo)
            v[k] = float(c + side * rng.uniform(span, 2 * span))
    return v


class VariantIterHybridEnv(IterHybridEnv):
    def __init__(self, variant: dict[str, float] | None = None, **kwargs):
        self.variant = {**NOMINAL, **(variant or {})}
        kwargs.setdefault("log_level", "warning")
        super().__init__(**kwargs)

    def _get_torax_config(self):
        cfg = super()._get_torax_config()
        return {**cfg, "config": apply_variant(CONFIG, self.variant)}


def load_eval_set(path: str = "configs/env/heldout.yaml") -> dict[str, list[dict[str, float]]]:
    from pathlib import Path

    import yaml

    return yaml.safe_load(Path(path).read_text())
