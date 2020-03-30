import torch
from torch.nn.functional import relu


def possqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(relu(x))


def channelwise_gram_matrix(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    x_flat = torch.flatten(x, 2)
    gram_matrix = torch.bmm(x_flat, x_flat.transpose(1, 2))
    if normalize:
        gram_matrix /= x_flat.size()[-1]
    return gram_matrix
