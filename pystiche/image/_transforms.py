from typing import Any, Optional, Sequence, Tuple, Union

import kornia

import torch
from torch import nn

__all__ = ["resize", "Affine", "Normalize", "Denormalize"]


def parse_align_corners(
    align_corners: Optional[bool], interpolation: str
) -> Optional[bool]:
    if align_corners is not None:
        return align_corners

    return (
        False
        if interpolation in ("linear", "bilinear", "bicubic", "trilinear")
        else None
    )


def resize(
    input: torch.Tensor,
    size: Union[int, Tuple[int, int]],
    align_corners: Optional[bool] = None,
    interpolation: str = "bilinear",
    side: str = "short",
) -> torch.Tensor:
    align_corners = parse_align_corners(align_corners, interpolation)
    return kornia.resize(
        input, size, interpolation=interpolation, align_corners=align_corners, side=side
    )


class _TensorToBufferMixin(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for name in dir(self):
            if name.startswith("_"):
                continue

            attr = getattr(self, name)
            if not isinstance(attr, torch.Tensor):
                continue

            delattr(self, name)
            self.register_buffer(name, attr)


class _AlignCornersMixin(nn.Module):
    def __init__(
        self,
        *args: Any,
        align_corners: Optional[bool] = None,
        interpolation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if interpolation is not None:
            kwargs["interpolation"] = interpolation
        else:
            interpolation = "bilinear"
        align_corners = parse_align_corners(align_corners, interpolation)
        super().__init__(*args, align_corners=align_corners, **kwargs)


class Affine(_TensorToBufferMixin, _AlignCornersMixin, kornia.Affine):
    _TENSOR_ATTRIBUTES = ("angle", "translation", "scale_factor", "shear", "center")

    def __init__(self, *args: Any, **kwargs: Any):
        for name in self._TENSOR_ATTRIBUTES:
            try:
                param = kwargs[name]
            except KeyError:
                continue

            kwargs[name] = self._maybe_to_tensor(param)

        super().__init__(*args, **kwargs)

    def _maybe_to_tensor(
        self, param: Optional[Union[float, Tuple[float, float]]]
    ) -> Optional[torch.Tensor]:
        if param is None:
            return None

        return torch.tensor(param).unsqueeze(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.size()[0]
        for name in self._TENSOR_ATTRIBUTES:
            self._maybe_batch_up(name, batch_size)

        return super().forward(input)

    def _maybe_batch_up(self, name: str, batch_size: int) -> None:
        attr = getattr(self, name)
        if attr is None:
            return

        setattr(self, name, attr.repeat(batch_size, *[1] * (attr.dim() - 1)))


class _NormalizeMixin(nn.Module):
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__(torch.tensor(mean), torch.tensor(std))


class Normalize(_TensorToBufferMixin, _NormalizeMixin, kornia.enhance.Normalize):
    pass


class Denormalize(_TensorToBufferMixin, _NormalizeMixin, kornia.enhance.Denormalize):
    pass
