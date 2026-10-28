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


class Paused(Exception):
    """Raised by an agent after its checkpoint is on disk, because a pause was requested."""


def _never() -> bool:
    return False


def _no_checkpoint(save: Callable[[Path], None]) -> None:
    return None


@dataclass
class TrainContext:
    budget: int
    seed: int
    run_dir: Path
    make_env: Callable[..., Any]
    tick: Callable[[int, dict | None], None]
    device: str = "cpu"
    extra: dict = field(default_factory=dict)
    resume: Path | None = None  # checkpoint directory to continue from (agent-specific files inside)
    should_pause: Callable[[], bool] = _never
    checkpoint_due: Callable[[], bool] = _never
    checkpoint: Callable[[Callable[[Path], None]], None] = _no_checkpoint

    def boundary(self, save: Callable[[Path], None]) -> None:
        """Call where the agent's whole state is consistent. `save(dir)` writes the agent's files.
        Checkpoints if one is due or a pause was requested; on pause, raises Paused once it is on disk."""
        if self.should_pause():
            self.checkpoint(save)
            raise Paused
        if self.checkpoint_due():
            self.checkpoint(save)


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
