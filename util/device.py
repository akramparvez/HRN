import os

import torch


def get_torch_device(prefer_mps=True):
    """Return the best available local torch device for macOS-first inference."""
    requested = os.environ.get("HRN_DEVICE", "").strip().lower()
    if requested:
        return torch.device(requested)
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path, device=None):
    if device is None:
        device = get_torch_device()
    return torch.load(path, map_location=device, weights_only=False)
