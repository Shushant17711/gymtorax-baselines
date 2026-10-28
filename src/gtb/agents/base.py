"""Agent interface used by scripts/train.py.

An agent is built from (cfg, run_dir). `learn(ctx)` consumes up to ctx.budget env steps, calling ctx.tick(n) after
every n env steps it takes (this drives logging and periodic evaluation). `save()` writes everything needed to
rebuild the policy to run_dir/model*. `policy_factory(run_dir)` (module level, picklable) rebuilds a policy in an
eval worker from the saved files.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainContext:
    budget: int
    seed: int
    run_dir: Path
    make_env: Callable[..., Any]
    tick: Callable[[int, dict | None], None]
    device: str = "cpu"
    extra: dict = field(default_factory=dict)


class Agent:
    name = "base"

    def __init__(self, cfg, run_dir: Path):
        self.cfg, self.run_dir = cfg, Path(run_dir)

    def learn(self, ctx: TrainContext) -> None:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    @staticmethod
    def policy_factory(run_dir: str):
        raise NotImplementedError
