"""Train one method/seed: uv run python scripts/train.py configs/agent/ppo.yaml seed=3 train.steps=1000"""
import argparse
import sys

from gtb.config import load_config
from gtb.train import EXIT_BUSY, EXIT_PAUSED, RunBusy, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--root", default="results")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    try:
        rd = train(load_config([a.base, a.config], a.overrides), root=a.root, overwrite=a.overwrite)
    except RunBusy as e:
        print(f"busy: another process is training {e}")
        sys.exit(EXIT_BUSY)
    if rd is None:
        sys.exit(EXIT_PAUSED)  # checkpoint saved; rerun the same command to continue
    print(rd)


if __name__ == "__main__":
    main()
