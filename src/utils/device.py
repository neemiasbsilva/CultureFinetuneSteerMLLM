"""Device selection utility — cuda > mps > cpu."""

import torch


def get_device() -> str:
    """Return the best available device for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
