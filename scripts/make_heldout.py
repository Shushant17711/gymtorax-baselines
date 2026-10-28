"""Write the frozen evaluation scenario list configs/env/heldout.yaml (nominal + 15 in-range + 5 OOD)."""
from pathlib import Path

import numpy as np
import yaml

from gtb.envs.variants import nominal, sample_variant

SEED = 20261028  # never used for training draws


def main():
    rng = np.random.default_rng(SEED)
    out = {"nominal": [nominal()],
           "heldout": [sample_variant(rng, "heldout") for _ in range(15)],
           "ood": [sample_variant(rng, "ood") for _ in range(5)]}
    out = {k: [{kk: round(x, 4) for kk, x in v.items()} for v in vs] for k, vs in out.items()}
    Path("configs/env/heldout.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
