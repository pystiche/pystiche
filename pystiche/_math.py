import torch
import pystiche
from pystiche.typing import Numeric


class _Safesqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.sqrt(input)
        output[input <= 0.0] = 0.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad = torch.rsqrt(input) / 2.0
        grad[input <= 0.0] = 0.0
        grad_input = grad * grad_output
        return grad_input


def safesqrt(x: torch.Tensor) -> torch.Tensor:
    return _Safesqrt.apply(x)


def msqrt(x: torch.Tensor) -> torch.Tensor:
    e, v = torch.symeig(x, eigenvectors=True)
    return torch.chain_matmul(v, torch.diag(e), v.t())


def channelwise_gram_matrix(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    x = pystiche.flatten_channelwise(x)
    G = torch.bmm(x, x.transpose(1, 2))
    if normalize:
        return G / x.size()[-1]
    else:
        return G


def examplewise_cosine_similarity(
    input: torch.Tensor, target: torch.Tensor, eps: Numeric = 1e-8
) -> torch.Tensor:
    input = pystiche.flatten_examplewise(input)
    input = input / torch.norm(input, dim=1, keepdim=True)

    target = pystiche.flatten_examplewise(target)
    target = target / torch.norm(target, dim=1, keepdim=True)

    return torch.clamp(torch.mm(input, target.t()), max=1.0 / eps)
