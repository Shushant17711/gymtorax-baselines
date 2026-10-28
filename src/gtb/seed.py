"""Global seeding for numpy, python and (if installed) torch."""
import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)
