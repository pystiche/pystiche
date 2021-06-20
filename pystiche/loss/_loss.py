from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple, Union, cast

import torch
from torch import nn

from pystiche import LossDict, enc

from .utils import apply_guide, match_batch_size

__all__ = [
    "Loss",
    "RegularizationLoss",
    "ComparisonLoss",
]


class Loss(nn.Module, ABC):
    def __init__(
        self,
        *,
        encoder: Optional[enc.Encoder] = None,
        input_guide: Optional[torch.Tensor] = None,
        score_weight: float = 1e0,
    ) -> None:
        super().__init__()

        self._encoder = encoder

        self._input_guide: Optional[torch.Tensor]
        self._input_enc_guide: Optional[torch.Tensor]
        if input_guide:
            self.set_input_guide(input_guide)
        else:
            for name in ("_input_guide", "_input_enc_guide"):
                self.register_buffer(
                    name,
                    None,  # type: ignore[arg-type]
                    persistent=False,
                )

        self.score_weight = score_weight

    @abstractmethod
    def forward(self, input_image: torch.Tensor) -> Union[torch.Tensor, LossDict]:
        pass

    @property
    def encoder(self) -> Optional[enc.Encoder]:
        return self._encoder

    @property
    def input_guide(self) -> Optional[torch.Tensor]:
        return self._input_guide

    def set_input_guide(self, guide: torch.Tensor) -> None:
        self.register_buffer("_input_guide", guide, persistent=False)
        self.register_buffer(
            "_input_enc_guide",
            self.encoder.propagate_guide(guide) if self.encoder else guide,
            persistent=False,
        )

    def _named_losses(self) -> Iterator[Tuple[str, "Loss"]]:
        for name, child in self.named_children():
            if isinstance(child, Loss):
                yield name, child

    def _losses(self) -> Iterator["Loss"]:
        for _, loss in self._named_losses():
            yield loss


class RegularizationLoss(Loss):
    def __init__(
        self,
        *,
        encoder: Optional[enc.Encoder] = None,
        input_guide: Optional[torch.Tensor] = None,
        score_weight: float = 1e0,
    ) -> None:
        super().__init__(
            encoder=encoder, input_guide=input_guide, score_weight=score_weight
        )

    def forward(self, image: torch.Tensor) -> Union[torch.Tensor, LossDict]:
        enc = self.encoder(image) if self.encoder else image
        if self._input_guide is not None:
            enc = apply_guide(
                enc,
                self._input_enc_guide
                if self._input_enc_guide is not None
                else self._input_guide,
            )

        repr = self.input_enc_to_repr(enc)
        return self.calculate_score(repr) * self.score_weight

    @abstractmethod
    def input_enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        pass


class ComparisonLoss(Loss):
    def __init__(
        self,
        *,
        encoder: Optional[enc.Encoder] = None,
        input_guide: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
        target_guide: Optional[torch.Tensor] = None,
        score_weight: float = 1e0,
    ) -> None:
        super().__init__(
            encoder=encoder, input_guide=input_guide, score_weight=score_weight
        )

        self._target_image: Optional[torch.Tensor]
        self._target_repr: Optional[torch.Tensor]
        self._target_guide: Optional[torch.Tensor]
        self._ctx: Optional[torch.Tensor]
        if target_image:
            self.set_target_image(target_image, guide=target_guide)
        else:
            for name in ("_target_image", "_target_repr", "_target_guide", "_ctx"):
                self.register_buffer(
                    name,
                    None,  # type: ignore[arg-type]
                    persistent=False,
                )

    def forward(self, input_image: torch.Tensor) -> Union[torch.Tensor, LossDict]:
        if self._target_image is None:
            raise RuntimeError

        input_enc = self.encoder(input_image) if self.encoder else input_image
        if self._input_guide is not None:
            input_enc = apply_guide(
                input_enc,
                self._input_enc_guide
                if self._input_enc_guide is not None
                else self._input_guide,
            )

        input_repr = self.input_enc_to_repr(input_enc, self._ctx)
        target_repr = match_batch_size(
            cast(torch.Tensor, self._target_repr), input_repr
        )
        return (
            self.calculate_score(input_repr, target_repr, self._ctx) * self.score_weight
        )

    @staticmethod
    def _match_batch_size(target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        # TODO
        return match_batch_size(target, input)

    @abstractmethod
    def input_enc_to_repr(
        self, enc: torch.Tensor, ctx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def target_enc_to_repr(
        self, enc: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

    @abstractmethod
    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pass

    @property
    def target_image(self) -> Optional[torch.Tensor]:
        return self._target_image

    @property
    def target_guide(self) -> Optional[torch.Tensor]:
        return self._target_guide

    def set_target_image(
        self, image: torch.Tensor, guide: Optional[torch.Tensor] = None
    ) -> None:
        self.register_buffer("_target_image", image, persistent=False)
        self.register_buffer("_target_guide", guide, persistent=False)

        with torch.no_grad():
            enc = self.encoder(image) if self.encoder else image

            if guide is not None:
                enc = apply_guide(
                    enc, self.encoder.propagate_guide(guide) if self.encoder else guide
                )

            repr, ctx = self.target_enc_to_repr(enc)

        self.register_buffer("_target_repr", repr, persistent=False)
        self.register_buffer("_ctx", ctx, persistent=False)
