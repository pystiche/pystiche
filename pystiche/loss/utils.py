import torch

__all__ = ["apply_guide", "match_batch_size"]


def apply_guide(image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
    return image * guide


def match_batch_size(target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    target_batch_size = target.size()[0]
    input_batch_size = input.size()[0]

    if target_batch_size == input_batch_size:
        return target

    if target_batch_size != 1:
        raise RuntimeError(
            f"If the batch size of the target != 1, "
            f"it has to match the batch size of the input. "
            f"Got {target_batch_size} != {input_batch_size}"
        )

    with torch.no_grad():
        return target.repeat(input_batch_size, *[1] * (target.dim() - 1))
