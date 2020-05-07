import torch
from torch.nn.functional import relu


def possqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(relu(x))


# FIXME: move this to pystiche_replication
# def msqrt(input: torch.Tensor) -> torch.Tensor:
#     e, v = torch.symeig(input, eigenvectors=True)
#     return torch.chain_matmul(v, torch.diag(e), v.t())
