import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Seed used for Python, NumPy, and PyTorch random
        number generators.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False