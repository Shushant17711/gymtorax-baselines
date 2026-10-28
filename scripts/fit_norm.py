"""Fit the frozen per-feature normalisation from random + scripted episodes -> configs/env/norm.json."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _episode(args):
    kind, seed = args
    from gymtorax import IterHybridAgent

    from gtb.envs.wrapper import FlatEnv, featurize

    env = FlatEnv()
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    agent = IterHybridAgent(env.env.action_space)
    feats, done = [featurize(env.last_state)], False
    while not done:
        if kind == "scripted":
            a = env.from_env_action(agent.act(env.last_state)) + rng.normal(0, 0.1 * (seed > 0), 7)
        else:
            a = rng.uniform(-1, 1, 7)
        _, _, te, tr, info = env.step(a)
        if not info["failed"]:
            feats.append(featurize(env.last_state))
        done = te or tr
    return np.stack(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="configs/env/norm.json")
    args = ap.parse_args()
    from gtb.envs.wrapper import FEATURE_NAMES

    jobs = [("random", s) for s in range(args.n)] + [("scripted", s) for s in range(args.n)]
    with ProcessPoolExecutor(args.workers) as ex:
        X = np.concatenate(list(ex.map(_episode, jobs)))
    mean, std = X.mean(0), X.std(0)
    std = np.maximum(std, 1e-3 * np.maximum(np.abs(mean), 1e-8) + 1e-8)  # std floor
    Path(args.out).write_text(json.dumps({"features": FEATURE_NAMES, "n_samples": len(X),
                                          "mean": mean.tolist(), "std": std.tolist()}, indent=1))
    print(f"wrote {args.out} from {len(X)} samples")


if __name__ == "__main__":
    main()
