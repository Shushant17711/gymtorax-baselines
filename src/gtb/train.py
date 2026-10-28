"""Training loop shared by all learned methods: run dir, metrics.jsonl, periodic held-out evaluation, final eval."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gtb.agents.base import TrainContext
from gtb.agents.registry import get_agent_cls
from gtb.config import Config, config_hash, save_config
from gtb.envs.train_env import make_train_env
from gtb.envs.variants import load_eval_set
from gtb.eval.metrics import iqm
from gtb.seed import seed_everything


def run_dir_for(cfg: Config, root: str = "results") -> Path:
    d = cfg.to_dict()
    seed = d.pop("seed")
    return Path(root) / cfg.agent.name / config_hash(d) / f"seed{seed}"


def summarise(suite: dict[str, list[dict]]) -> dict:
    out = {}
    for split, recs in suite.items():
        r = np.array([x["return"] for x in recs])
        out[split] = {"iqm": iqm(r), "mean": float(r.mean()), "n": len(r),
                      "fail_rate": float(np.mean([x["failed"] for x in recs])),
                      "frac_q_min_lt_1": float(np.mean([x["frac_q_min_lt_1"] for x in recs])),
                      "frac_q95_lt_3": float(np.mean([x["frac_q95_lt_3"] for x in recs]))}
    return out


def evaluate_run(run_dir: Path, cfg: Config, splits=None, workers: int | None = None) -> dict:
    from gtb.eval.rollout import eval_suite

    es = load_eval_set(cfg.train.eval_set)
    if splits:
        es = {k: es[k] for k in splits}
    factory = get_agent_cls(cfg.agent.name).policy_factory(str(run_dir))
    suite = eval_suite(factory, es, workers=workers or cfg.train.eval_workers, norm_path=cfg.env.norm)
    return {"summary": summarise(suite), "episodes": suite}


def _returns(ev: dict) -> dict[str, list[float]]:
    return {k: [float(r["return"]) for r in v] for k, v in ev["episodes"].items()}


def train(cfg: Config, root: str = "results", overwrite: bool = False) -> Path:
    rd = run_dir_for(cfg, root)
    if (rd / "eval.json").exists() and not overwrite:
        print(f"done already: {rd}")
        return rd
    rd.mkdir(parents=True, exist_ok=True)
    save_config(cfg, rd / "config.yaml")
    seed_everything(int(cfg.seed))
    agent = get_agent_cls(cfg.agent.name)(cfg, rd)
    metrics = (rd / "metrics.jsonl").open("w")
    state = {"steps": 0, "next_eval": int(cfg.train.eval_every), "t0": time.time()}

    def log(rec: dict) -> None:
        metrics.write(json.dumps({"steps": state["steps"], "wall": round(time.time() - state["t0"], 1), **rec}) + "\n")
        metrics.flush()

    def tick(n: int, rec: dict | None = None) -> None:
        state["steps"] += n
        if rec:
            log(rec)
        if state["steps"] >= state["next_eval"] and state["steps"] < cfg.train.steps:
            state["next_eval"] += int(cfg.train.eval_every)
            agent.save()
            ev = evaluate_run(rd, cfg, splits=list(cfg.train.periodic_splits))
            log({"eval": ev["summary"], "returns": _returns(ev)})

    def make_env(seed: int = 0, split: str = "train"):
        return make_train_env(seed=seed, split=split, norm_path=cfg.env.norm)

    ctx = TrainContext(budget=int(cfg.train.steps), seed=int(cfg.seed), run_dir=rd, make_env=make_env, tick=tick,
                       device=cfg.get("device", "cpu"))
    agent.learn(ctx)
    agent.save()
    ev = evaluate_run(rd, cfg)
    log({"eval": ev["summary"], "returns": _returns(ev), "final": True})
    metrics.close()
    (rd / "eval.json").write_text(json.dumps({"env_steps": state["steps"], "wall_s": time.time() - state["t0"],
                                              **ev}, default=float))
    return rd
