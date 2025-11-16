import torch
import numpy as np
import random

def set_seed(seed: int):
    """
    Sets the random seed for numpy, random, and torch for reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: for full determinism, but can slow down training
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Set all random seeds to {seed}.")