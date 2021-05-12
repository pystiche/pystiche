from abc import abstractmethod
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch

import pystiche
from pystiche.enc import Encoder

__all__ = [
    "Operator",
    "RegularizationOperator",
    "ComparisonOperator",
    "PixelOperator",
    "EncodingOperator",
    "PixelRegularizationOperator",
    "EncodingRegularizationOperator",
    "PixelComparisonOperator",
    "EncodingComparisonOperator",
]


class Operator(pystiche.Module):
    r"""Abstract base class for all operators. If called, invokes
    :meth:`pystiche.ops.Operator.process_input_image` and applies ``score_weight`` to
    the result.

    Args:
        score_weight: Score weight of the operator. Defaults to ``1.0``.
    """

    def __init__(self, score_weight: float = 1e0,) -> None:
        super().__init__()
        self.score_weight = score_weight

    input_guide: torch.Tensor

    def set_input_guide(self, guide: torch.Tensor) -> None:
        r"""Set input guide.

        Args:
            guide: Input guide of shape :math:`1 \times 1 \times H \times W`.
        """
        self.register_buffer("input_guide", guide, persistent=False)

    @property
    def has_input_guide(self) -> bool:
        return "input_guide" in self._buffers

    @staticmethod
    def apply_guide(image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        r"""Apply a guide to an image.

        Args:
            image: Image of shape :math:`B \times C \times H \times W`.
            guide: Guide of shape :math:`1 \times 1 \times H \times W`.
        """
        return image * guide

    def forward(
        self, input_image: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.LossDict]:
        return self.process_input_image(input_image) * self.score_weight

    @abstractmethod
    def process_input_image(
        self, image: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.LossDict]:
        r"""Defines the computation performed with every call.

        Args:
            image: Input image of shape :math:`B \times C \times H \times W`.
        """
        pass

    def named_operators(
        self, recurse: bool = False,
    ) -> Iterator[Tuple[str, "Operator"]]:
        if recurse:
            iterator = self.named_modules()
            # First module is always self so dismiss that
            next(iterator)
        else:
            iterator = self.named_children()
        for name, child in iterator:
            if isinstance(child, Operator):
                yield name, child

    def operators(self, recurse: bool = False) -> Iterator["Operator"]:
        for _, op in self.named_operators(recurse=recurse):
            yield op

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["score_weight"] = f"{self.score_weight:g}"
        return dct


class RegularizationOperator(Operator):
    r"""Abstract base class for all regularization operators."""

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class ComparisonOperator(Operator):
    r"""Abstract base class for all comparison operators."""
    target_guide: torch.Tensor
    target_image: torch.Tensor
    target_repr: torch.Tensor
    ctx: Optional[torch.Tensor]

    @abstractmethod
    def set_target_guide(self, guide: torch.Tensor, recalc_repr: bool = True) -> None:
        r"""Set a target guide and optionally recalculate the target representation.

        Args:
            guide: Input guide of shape :math:`1 \times 1 \times H \times W`.
            recalc_repr: If ``True``, recalculates :meth:`.target_enc_to_repr`.
                Defaults to ``True``.

                .. note::

                    Set this to ``False`` if the shape of ``guide`` and the shape of a
                    previously set ``target_image`` do not match.
        """
        pass

    @property
    def has_target_guide(self) -> bool:
        return "target_guide" in self._buffers

    @abstractmethod
    def set_target_image(self, image: torch.Tensor) -> None:
        r"""Set a target image and calculate its representation.

        Args:
            image: Target image of shape :math:`B \times C \times H \times W`.
        """
        pass

    @property
    def has_target_image(self) -> bool:
        return "target_image" in self._buffers

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def _match_batch_sizes(target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
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


class PixelOperator(Operator):
    r"""Abstract base class for all operators working in the pixel space."""

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class EncodingOperator(Operator):
    r"""Abstract base class for all operators working in an encoded space."""

    encoder: Encoder

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["encoder"] = self.encoder
        return dct

    def __repr__(self) -> str:
        return self._build_repr(
            named_children=[
                (name, child)
                for name, child in self.named_children()
                if child is not self.encoder
            ]
        )


class PixelRegularizationOperator(PixelOperator, RegularizationOperator):
    r"""Abstract base class for all regularization operators working in the pixel space.

    Args:
        score_weight: Score weight of the operator. Defaults to ``1.0``.
    """

    def __init__(self, score_weight: float = 1.0):
        super().__init__(score_weight=score_weight,)

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.has_input_guide:
            image = self.apply_guide(image, self.input_guide)
        input_repr = self.input_image_to_repr(image)
        return self.calculate_score(input_repr)

    @abstractmethod
    def input_image_to_repr(self, image: torch.Tensor) -> torch.Tensor:
        r"""Calculate the input representation from the input image.

        .. note::

            This method has to be overwritten in every subclass.

        Args:
            image: Input image of shape :math:`B \times C \times H \times W`.
        """
        pass

    @abstractmethod
    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        r"""Calculate the operator score from the input representation.

        .. note::

            This method has to be overwritten in every subclass.

        Args:
            input_repr: Input representation.
        """
        pass


class EncodingRegularizationOperator(EncodingOperator, RegularizationOperator):
    r"""Abstract base class for all regularization operators working in an encoded
    space.

    Args:
        encoder: Encoder that is used to encode the target and input images.
        score_weight: Score weight of the operator. Defaults to ``1.0``.
    """

    def __init__(self, encoder: Encoder, score_weight: float = 1.0):
        super().__init__(score_weight=score_weight)
        self.encoder = encoder

    def set_input_guide(self, guide: torch.Tensor) -> None:
        super().set_input_guide(guide)
        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer("input_enc_guide", enc_guide, persistent=False)

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        input_repr = self.input_image_to_repr(image)
        return self.calculate_score(input_repr)

    def input_image_to_repr(self, image: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(image)
        if self.has_input_guide:
            enc = self.apply_guide(enc, self.input_guide)
        return self.input_enc_to_repr(enc)

    @abstractmethod
    def input_enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        r"""Calculate the input representation from the encoded input image.

        .. note::

            This method has to be overwritten in every subclass.

        Args:
            enc: Encoded input image of shape :math:`B \times C \times H \times W`.
        """
        pass

    @abstractmethod
    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        r"""Calculate the operator score from the input representation.

        .. note::

            This method has to be overwritten in every subclass.

        Args:
            input_repr: Input representation.
        """
        pass


class PixelComparisonOperator(PixelOperator, ComparisonOperator):
    r"""Abstract base class for all comparison operators working in the pixel space.

    Args:
        score_weight: Score weight of the operator. Defaults to ``1.0``.

    Raises:
        RuntimeError: If called without setting a target image first.
    """

    def __init__(self, score_weight: float = 1e0):
        super().__init__(score_weight=score_weight)

    def set_target_guide(self, guide: torch.Tensor, recalc_repr: bool = True) -> None:
        self.register_buffer("target_guide", guide, persistent=False)
        if recalc_repr and self.has_target_image:
            self.set_target_image(self.target_image)

    def set_target_image(self, image: torch.Tensor) -> None:
        self.register_buffer("target_image", image, persistent=False)
        with torch.no_grad():
            if self.has_target_guide:
                image = self.apply_guide(image, self.target_guide)
            repr, ctx = self.target_image_to_repr(image)
        self.register_buffer("target_repr", repr, persistent=False)
        if ctx is not None:
            self.register_buffer("ctx", ctx, persistent=False)
        else:
            self.ctx = None

    @abstractmethod
    def target_image_to_repr(
        self, image: torch.Tensor
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor],
    ]:
        r"""Calculate the target representation and context information from the
        target image.

        .. note::

            This method has to be overwritten in every subclass.

        Args:
            image: Target image of shape :math:`B \times C \times H \times W`.
        """
        pass

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self.has_target_image:
            msg = "Cannot process an input image before a target image has been set."
            raise RuntimeError(msg)
        target_repr, ctx = self.target_repr, self.ctx

        if self.has_input_guide:
            image = self.apply_guide(image, self.input_guide)
        input_repr = self.input_image_to_repr(image, ctx)

        target_repr = self._match_batch_sizes(target_repr, input_repr)
        return self.calculate_score(input_repr, target_repr, ctx)

    @abstractmethod
    def input_image_to_repr(
        self, image: torch.Tensor, ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        r"""Calculate the input representation from the input image optionally using
        the context information.

        .. note::

            This method has to be overwritten in every subclass.

        Args:
            image: Input image of shape :math:`B \times C \times H \times W`.
            ctx: Optional context information.
        """
        pass

    @abstractmethod
    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        r"""Calculate the operator score from the input and target representation
        optionally using the context information.

        .. note::

            This method has to be overwritten in every subclass.

        Args:
            input_repr: Input representation.
            target_repr: Target representation.
            ctx: Optional context information.
        """
        pass


class EncodingComparisonOperator(EncodingOperator, ComparisonOperator):
    r"""Abstract base class for all comparison operators working in an encoded space.

    Args:
        encoder: Encoder that is used to encode the target and input images.
        score_weight: Score weight of the operator. Defaults to ``1.0``.

    Raises:
        RuntimeError: If called without setting a target image first.
    """
    target_enc_guide: torch.Tensor
    input_enc_guide: torch.Tensor

    def __init__(self, encoder: Encoder, score_weight: float = 1.0):
        super().__init__(score_weight=score_weight)
        self.encoder = encoder

    def set_target_guide(self, guide: torch.Tensor, recalc_repr: bool = True) -> None:

        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer("target_guide", guide, persistent=False)
        self.register_buffer("target_enc_guide", enc_guide, persistent=False)
        if recalc_repr and self.has_target_image:
            self.set_target_image(self.target_image)

    def set_target_image(self, image: torch.Tensor) -> None:

        with torch.no_grad():
            repr, ctx = self.target_image_to_repr(image)
        self.register_buffer("target_image", image, persistent=False)
        self.register_buffer("target_repr", repr, persistent=False)
        if ctx is not None:
            self.register_buffer("ctx", ctx, persistent=False)
        else:
            self.ctx = None

    def target_image_to_repr(
        self, image: torch.Tensor
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor],
    ]:
        enc = self.encoder(image)
        if self.has_target_guide:
            enc = self.apply_guide(enc, self.target_enc_guide)
        return self.target_enc_to_repr(enc)

    @abstractmethod
    def target_enc_to_repr(
        self, enc: torch.Tensor
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor],
    ]:
        r"""Calculate the target representation and context information from the
        encoded target image.

        .. note::

            This method has to be overwritten in every subclass.

        Args:
            enc: Encoded target image of shape :math:`B \times C \times H \times W`.
        """

    def set_input_guide(self, guide: torch.Tensor) -> None:
        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer("input_guide", guide, persistent=False)
        self.register_buffer("input_enc_guide", enc_guide, persistent=False)

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self.has_target_image:
            msg = "Cannot process an input image before a target image has been set."
            raise RuntimeError(msg)
        target_repr, ctx = self.target_repr, self.ctx
        input_repr = self.input_image_to_repr(image, ctx)

        target_repr = self._match_batch_sizes(target_repr, input_repr)
        return self.calculate_score(input_repr, target_repr, ctx)

    def input_image_to_repr(
        self, image: torch.Tensor, ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        enc = self.encoder(image)
        if self.has_input_guide:
            enc = self.apply_guide(enc, self.input_enc_guide)
        return self.input_enc_to_repr(enc, ctx)

    @abstractmethod
    def input_enc_to_repr(
        self, enc: torch.Tensor, ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        r"""Calculate the input representation from the encoded input image optionally
        using the context information.

        .. note::

            This method has to be overwritten in every subclass.

        Args:
            enc: Encoded input image of shape :math:`B \times C \times H \times W`.
            ctx: Optional context information.
        """
        pass

    @abstractmethod
    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        r"""Calculate the operator score from the input and target representation
        optionally using the context information.

        .. note::

            This method has to be overwritten in every subclass.

        Args:
            input_repr: Input representation.
            target_repr: Target representation.
            ctx: Optional context information.
        """
        pass
