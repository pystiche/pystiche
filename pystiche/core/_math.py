import torch
from torch.nn.functional import relu


def possqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(relu(x))
