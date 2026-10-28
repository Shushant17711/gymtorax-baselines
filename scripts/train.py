"""Train one method/seed: uv run python scripts/train.py configs/agent/ppo.yaml seed=3 train.steps=1000"""
import argparse

from gtb.config import load_config
from gtb.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--root", default="results")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    print(train(load_config([a.base, a.config], a.overrides), root=a.root, overwrite=a.overwrite))


if __name__ == "__main__":
    main()
