"""Dummy agent for pipeline tests: steps random actions, 'learns' nothing, evaluates as a seeded random policy."""
from __future__ import annotations

import json
from functools import partial

from gtb.agents.base import Agent, TrainContext
from gtb.agents.reference import RandomPolicy


class DummyAgent(Agent):
    name = "dummy"

    def learn(self, ctx: TrainContext) -> None:
        env = ctx.make_env(seed=ctx.seed)
        env.reset(seed=ctx.seed)
        rng = env.action_space
        rng.seed(ctx.seed)
        ep_ret, steps = 0.0, 0
        while steps < ctx.budget:
            _, r, te, tr, _ = env.step(rng.sample())
            ep_ret += r
            steps += 1
            done = te or tr
            ctx.tick(1, {"episode_return": ep_ret} if done else None)
            if done:
                env.reset()
                ep_ret = 0.0
        env.close()

    def save(self) -> None:
        (self.run_dir / "model.json").write_text(json.dumps({"seed": int(self.cfg.seed)}))

    @staticmethod
    def policy_factory(run_dir: str):
        return partial(_load, run_dir)


def _load(run_dir: str):
    from pathlib import Path

    return RandomPolicy(json.loads((Path(run_dir) / "model.json").read_text())["seed"])
