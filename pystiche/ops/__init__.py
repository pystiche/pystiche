# type: ignore

import functools
import re
import warnings
from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch

import pystiche
from pystiche import enc, loss
from pystiche.misc import build_deprecation_message

_PATTERN = re.compile(re.escape("<class 'pystiche.loss."))


def __op_init__(
    self, *args: Any, __old_name__: str, __new_name__: str, **kwargs: Any
) -> None:
    msg = build_deprecation_message(
        f"The class pystiche.ops.{__old_name__}",
        "1.0",
        info=(
            f"It was renamed and moved to pystiche.loss.{__new_name__}. "
            f"See https://github.com/pystiche/pystiche/issues/436 for details"
        ),
    )
    warnings.warn(msg)
    for super_cls in type(self).__mro__:
        if _PATTERN.match(str(super_cls)):
            break
    else:
        raise RuntimeError
    super_cls.__init__(self, *args, **kwargs)


def _input_guide(self: loss.Loss) -> torch.Tensor:
    return self._input_guide


def _has_input_guide(self: loss.Loss):
    return self._input_guide is not None


def _add_operator_attributes_and_methods(op_cls):
    op_cls.input_guide = property(_input_guide)
    op_cls.has_input_guide = property(_has_input_guide)
    op_cls.apply_guide = staticmethod(loss.utils.apply_guide)
    return op_cls


def _target_guide(self: loss.Loss) -> torch.Tensor:
    return self._target_guide


def _has_target_guide(self: loss.Loss):
    return self._target_guide is not None


def _target_image(self: loss.Loss) -> torch.Tensor:
    return self._target_image


def _has_target_image(self: loss.Loss):
    return self._target_image is not None


def _set_target_guide(
    self: loss.ComparisonLoss, guide: torch.Tensor, recalc_repr: bool = True
) -> None:
    self.set_target_image(image=None, guide=guide, _recalc_repr=recalc_repr)


def _set_target_image(
    self: loss.ComparisonLoss,
    image: Optional[torch.Tensor],
    guide: Optional[torch.Tensor] = None,
    _recalc_repr: bool = True,
) -> None:
    set_target_image = functools.partial(loss.ComparisonLoss.set_target_image, self)

    if guide is not None and image is not None:
        set_target_image(image, guide=guide)
    elif image is None:
        if _recalc_repr and self._target_image is not None:
            set_target_image(self._target_image, guide=guide)
        else:
            self.register_buffer("_target_guide", guide, persistent=False)
    elif guide is None:
        set_target_image(image, guide=self._target_guide)


def _target_repr(self: loss.ComparisonLoss) -> Optional[torch.Tensor]:
    return self._target_repr


def _ctx(self: loss.ComparisonLoss) -> Optional[torch.Tensor]:
    return self._ctx


def _target_enc_guide(self: loss.ComparisonLoss) -> torch.Tensor:
    assert self.encoder
    return self.encoder.propagate_guide(self.target_guide)


def _input_enc_guide(self: loss.ComparisonLoss) -> torch.Tensor:
    assert self.encoder
    return self.encoder.propagate_guide(self.input_guide)


def _add_comparison_operator_attributes_and_methods(comparison_op_cls):
    comparison_op_cls.target_guide = property(_target_guide)
    comparison_op_cls.has_target_guide = property(_has_target_guide)
    comparison_op_cls.set_target_guide = _set_target_guide
    comparison_op_cls.target_image = property(_target_image)
    comparison_op_cls.has_target_image = property(_has_target_image)
    comparison_op_cls.set_target_image = _set_target_image
    comparison_op_cls.target_repr = property(_target_repr)
    comparison_op_cls.ctx = property(_ctx)
    comparison_op_cls.target_enc_guide = property(_target_enc_guide)
    comparison_op_cls.input_enc_guide = property(_input_enc_guide)
    return comparison_op_cls


def _regularization_input_image_to_repr(
    self: loss.RegularizationLoss, image: torch.Tensor
) -> torch.Tensor:
    pass


def _regularization_input_enc_to_repr(
    self: loss.RegularizationLoss, enc: torch.Tensor
) -> torch.Tensor:
    assert not self.encoder
    return self.input_image_to_repr(enc)


def _add_pixel_regularization_operator_methods(pixel_regularization_op_cls):
    pixel_regularization_op_cls.input_image_to_repr = abstractmethod(
        _regularization_input_image_to_repr
    )
    pixel_regularization_op_cls.input_enc_to_repr = _regularization_input_enc_to_repr
    return pixel_regularization_op_cls


def _comparison_input_image_to_repr(
    self: loss.RegularizationLoss, image: torch.Tensor, ctx: Optional[torch.Tensor],
) -> torch.Tensor:
    pass


def _comparison_input_enc_to_repr(
    self: loss.Loss, enc: torch.Tensor, ctx: Optional[torch.Tensor]
) -> torch.Tensor:
    assert not self.encoder
    return self.input_image_to_repr(enc, ctx)


def _target_image_to_repr(
    self: loss.Loss, image: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    pass


def _target_enc_to_repr(
    self: loss.Loss, enc: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    assert not self.encoder
    return self.target_image_to_repr(enc)


def _add_pixel_comparison_operator_methods(pixel_comparison_op_cls):
    pixel_comparison_op_cls.input_image_to_repr = abstractmethod(
        _comparison_input_image_to_repr
    )
    pixel_comparison_op_cls.input_enc_to_repr = _comparison_input_enc_to_repr
    pixel_comparison_op_cls.target_image_to_repr = abstractmethod(_target_image_to_repr)
    pixel_comparison_op_cls.target_enc_to_repr = _target_enc_to_repr
    return pixel_comparison_op_cls


@_add_operator_attributes_and_methods
class Operator(loss.Loss):
    def forward(self, image: torch.Tensor):
        return self.process_input_image(image) * self.score_weight

    @abstractmethod
    def process_input_image(self, image: torch.Tensor):
        pass


@_add_operator_attributes_and_methods
class RegularizationOperator(loss.RegularizationLoss):
    pass


@_add_operator_attributes_and_methods
@_add_comparison_operator_attributes_and_methods
class ComparisonOperator(loss.ComparisonLoss):
    pass


@_add_operator_attributes_and_methods
class PixelOperator(loss.Loss):
    pass


@_add_operator_attributes_and_methods
class EncodingOperator(loss.Loss):
    pass


@_add_operator_attributes_and_methods
@_add_pixel_regularization_operator_methods
class PixelRegularizationOperator(PixelOperator, RegularizationOperator):
    def __init__(self, score_weight: float = 1e0) -> None:
        super().__init__(score_weight=score_weight)


@_add_operator_attributes_and_methods
class EncodingRegularizationOperator(EncodingOperator, RegularizationOperator):
    def __init__(self, encoder: enc.Encoder, score_weight: float = 1e0) -> None:
        super().__init__(encoder=encoder, score_weight=score_weight)


@_add_operator_attributes_and_methods
@_add_comparison_operator_attributes_and_methods
@_add_pixel_comparison_operator_methods
class PixelComparisonOperator(PixelOperator, ComparisonOperator):
    def __init__(self, score_weight: float = 1e0) -> None:
        super().__init__(score_weight=score_weight)


@_add_operator_attributes_and_methods
@_add_comparison_operator_attributes_and_methods
class EncodingComparisonOperator(EncodingOperator, ComparisonOperator):
    def __init__(self, encoder: enc.Encoder, score_weight: float = 1e0) -> None:
        super().__init__(encoder=encoder, score_weight=score_weight)


def _get_image_or_guide(
    self: loss.Loss, attr: str, comparison_only: bool = False
) -> torch.Tensor:
    images_or_guides: List[torch.Tensor] = []
    for op in self._losses():
        if comparison_only and not isinstance(op, loss.ComparisonLoss):
            continue

        try:
            image_or_guide = getattr(op, attr)
        except AttributeError:
            continue

        if image_or_guide is not None:
            images_or_guides.append(image_or_guide)

    if not images_or_guides:
        raise RuntimeError(f"No immediate children has a {attr}.")

    image_or_guide = images_or_guides[0]
    key = pystiche.TensorKey(image_or_guide)

    if not all(key == other for other in images_or_guides[1:]):
        raise RuntimeError(f"The immediate children have non-matching {attr}")

    return image_or_guide


def _get_target_guide(self: loss.Loss) -> torch.Tensor:
    return self._get_image_or_guide("target_guide", comparison_only=True)


def _get_target_image(self: loss.Loss) -> torch.Tensor:
    return self._get_image_or_guide("target_image", comparison_only=True)


def _get_input_guide(self: loss.Loss) -> torch.Tensor:
    return self._get_image_or_guide("input_guide")


def _set_image_or_guide(
    self,
    image_or_guide: torch.Tensor,
    attr: str,
    comparison_only: bool = False,
    **kwargs: Any,
) -> None:
    for op in self._losses():
        if comparison_only and not isinstance(op, loss.ComparisonLoss):
            continue

        setter = getattr(op, f"set_{attr}")
        setter(image_or_guide, **kwargs)


def _container_set_target_guide(
    self, guide: torch.Tensor, recalc_repr: bool = True
) -> None:
    self._set_image_or_guide(
        guide, "target_guide", comparison_only=True, recalc_repr=recalc_repr
    )


def _container_set_target_image(
    self, image: torch.Tensor, guide: Optional[torch.Tensor] = None,
) -> None:
    if guide is not None:
        self.set_target_guide(guide, recalc_repr=False)
    self._set_image_or_guide(image, "target_image", comparison_only=True)


def _container_set_input_guide(self, guide: torch.Tensor) -> None:
    self._set_image_or_guide(guide, "input_guide")


def _add_container_methods(op_container_cls):
    op_container_cls._get_image_or_guide = _get_image_or_guide
    op_container_cls.get_target_guide = _get_target_guide
    op_container_cls.get_target_image = _get_target_image
    op_container_cls.get_input_guide = _get_input_guide
    op_container_cls._set_image_or_guide = _set_image_or_guide
    op_container_cls.set_target_guide = _container_set_target_guide
    op_container_cls.set_target_image = _container_set_target_image
    op_container_cls.set_input_guide = _container_set_input_guide
    return op_container_cls


@_add_operator_attributes_and_methods
@_add_container_methods
class SameOperatorContainer(loss.SameTypeLossContainer):
    def __init__(
        self, *args, op_weights: Union[str, Sequence[float]] = "sum", **kwargs
    ) -> None:
        super().__init__(*args, loss_weights=op_weights, **kwargs)


class MultiRegionOperator(loss.MultiRegionLoss):
    def set_regional_target_guide(self, region: str, guide: torch.Tensor) -> None:
        getattr(self, region).set_target_guide(guide)


_NAME_MAP = {"SameTypeLossContainer": "SameOperatorContainer"}

__all__ = []
for loss_name in dir(loss):
    if loss_name.startswith("_") or "Loss" not in loss_name:
        continue

    loss_cls = getattr(loss, loss_name)
    if not isinstance(loss_cls, type):
        continue

    op_name = _NAME_MAP.get(loss_name, loss_name.replace("Loss", "Operator"))
    if op_name in globals():
        continue

    op_cls = type(
        op_name,
        (loss_cls,),
        dict(
            __init__=functools.partialmethod(
                __op_init__, __old_name__=op_name, __new_name__=loss_name
            )
        ),
    )

    op_cls = _add_operator_attributes_and_methods(op_cls)
    if issubclass(loss_cls, loss.ComparisonLoss):
        op_cls = _add_comparison_operator_attributes_and_methods(op_cls)
    elif issubclass(loss_cls, loss.LossContainer):
        op_cls = _add_container_methods(op_cls)

    __all__.append(op_name)
    globals()[op_name] = op_cls
