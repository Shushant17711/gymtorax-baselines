"""Training loop shared by all learned methods: run dir, metrics.jsonl, periodic held-out evaluation, final eval."""
from __future__ import annotations

import json
import os
import shutil
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

# Settings that change resource use but not results; excluded from the run hash.
RESOURCE_KEYS = (("train", "eval_workers"), ("agent", "pool_size"), ("agent", "torch_threads"))


def run_dir_for(cfg: Config, root: str = "results") -> Path:
    d = cfg.to_dict()
    seed = d.pop("seed")
    for sec, k in RESOURCE_KEYS:
        d.get(sec, {}).pop(k, None)
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
    from gtb.envs.wrapper import set_q_cap

    set_q_cap(cfg.env.get("q_cap"))
    from gtb.eval.rollout import eval_suite

    es = load_eval_set(cfg.train.eval_set)
    if splits:
        es = {k: es[k] for k in splits}
    factory = get_agent_cls(cfg.agent.name).policy_factory(str(run_dir))
    suite = eval_suite(factory, es, workers=workers or cfg.train.eval_workers, norm_path=cfg.env.norm)
    return {"summary": summarise(suite), "episodes": suite}


def _returns(ev: dict) -> dict[str, list[float]]:
    return {k: [float(r["return"]) for r in v] for k, v in ev["episodes"].items()}


EXIT_PAUSED, EXIT_BUSY = 75, 73


class RunBusy(Exception):
    """Another process holds this run's lock."""


def pause_file() -> Path:
    """Creating this file pauses every training job (checked at each agent's safe points)."""
    return Path(os.environ.get("GTB_PAUSE_FILE", "PAUSE"))


def _ckpt_prepare(rd: Path) -> Path | None:
    """Return rd/ckpt if a complete checkpoint exists, recovering from a crash in the middle of a swap.

    A checkpoint is complete once its progress.json exists (written last). Newest first: ckpt.new, ckpt, ckpt.old."""
    c, new, old = rd / "ckpt", rd / "ckpt.new", rd / "ckpt.old"
    found = next((d for d in (new, c, old) if (d / "progress.json").exists()), None)
    if found is not None and found != c:
        shutil.rmtree(c, ignore_errors=True)
        found.rename(c)
    for d in (new, old):
        shutil.rmtree(d, ignore_errors=True)
    return c if found is not None else None


def _ckpt_commit(rd: Path, new: Path) -> None:
    c, old = rd / "ckpt", rd / "ckpt.old"
    shutil.rmtree(old, ignore_errors=True)
    if c.exists():
        c.rename(old)
    new.rename(c)
    shutil.rmtree(old, ignore_errors=True)


def train(cfg: Config, root: str = "results", overwrite: bool = False) -> Path | None:
    """Train one run. Resumes from the run's checkpoint if there is one. Returns None if paused."""
    import fcntl

    from gtb.agents.base import Paused
    from gtb.envs.wrapper import set_q_cap

    set_q_cap(cfg.env.get("q_cap"))  # before any env or worker pool exists
    rd = run_dir_for(cfg, root)
    if (rd / "eval.json").exists() and not overwrite:
        print(f"done already: {rd}")
        return rd
    rd.mkdir(parents=True, exist_ok=True)
    lock = (rd / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released when this process exits
    except BlockingIOError:
        raise RunBusy(str(rd)) from None
    if overwrite:
        for d in ("ckpt", "ckpt.new", "ckpt.old"):
            shutil.rmtree(rd / d, ignore_errors=True)
    save_config(cfg, rd / "config.yaml")
    seed_everything(int(cfg.seed))
    agent = get_agent_cls(cfg.agent.name)(cfg, rd)
    mpath = rd / "metrics.jsonl"
    ck = _ckpt_prepare(rd)
    if ck is not None:
        prog = json.loads((ck / "progress.json").read_text())
        mpath.touch()
        with mpath.open("r+b") as f:
            f.truncate(prog["metrics_bytes"])  # drop anything logged after the checkpoint (it is redone)
        metrics = mpath.open("a")
        state = {"steps": prog["steps"], "next_eval": prog["next_eval"], "t0": time.time() - prog["wall"]}
    else:
        metrics = mpath.open("w")
        state = {"steps": 0, "next_eval": int(cfg.train.eval_every), "t0": time.time()}
    state["last_ckpt"] = time.time()
    ckpt_minutes = float(os.environ.get("GTB_CKPT_MINUTES", "5"))
    pause_at = int(os.environ.get("GTB_PAUSE_AT_STEPS", "0"))  # testing aid: pause at the first safe point >= N

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

    def checkpoint(save) -> None:
        new = rd / "ckpt.new"
        shutil.rmtree(new, ignore_errors=True)
        new.mkdir()
        save(new)
        metrics.flush()
        os.sync()
        (new / "progress.json").write_text(json.dumps({
            "steps": state["steps"], "next_eval": state["next_eval"], "wall": time.time() - state["t0"],
            "metrics_bytes": mpath.stat().st_size, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}))
        os.sync()
        _ckpt_commit(rd, new)
        state["last_ckpt"] = time.time()

    def should_pause() -> bool:
        return pause_file().exists() or (pause_at > 0 and state["steps"] >= pause_at)

    if ck is not None:
        log({"resumed": True})
        print(f"resuming {rd} at {state['steps']} steps", flush=True)
    ctx = TrainContext(budget=int(cfg.train.steps), seed=int(cfg.seed), run_dir=rd, make_env=make_env, tick=tick,
                       device=cfg.get("device", "cpu"), resume=ck, should_pause=should_pause,
                       checkpoint_due=lambda: time.time() - state["last_ckpt"] > 60 * ckpt_minutes,
                       checkpoint=checkpoint)
    try:
        agent.learn(ctx)
    except Paused:
        metrics.close()
        print(f"paused at {state['steps']} steps: {rd}", flush=True)
        return None
    agent.save()
    ev = evaluate_run(rd, cfg)
    log({"eval": ev["summary"], "returns": _returns(ev), "final": True})
    metrics.close()
    (rd / "eval.json").write_text(json.dumps({"env_steps": state["steps"], "wall_s": time.time() - state["t0"],
                                              **ev}, default=float))
    for d in ("ckpt", "ckpt.new", "ckpt.old"):  # finished: the checkpoint is no longer needed
        shutil.rmtree(rd / d, ignore_errors=True)
    return rd
