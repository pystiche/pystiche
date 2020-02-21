from typing import Optional
import torch
from torch import nn

__all__ = ["ResidualBlock"]


class ResidualBlock(nn.Module):
    def __init__(self, residual: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.residual = residual

        class Identity(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        if shortcut is None:
            shortcut = Identity()
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor):
        return self.residual(x) + self.shortcut(x)
