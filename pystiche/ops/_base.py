from abc import abstractmethod
from typing import Any, Union, Sequence, Tuple, Dict
from collections import OrderedDict
import torch
import pystiche
from pystiche.typing import Numeric
from pystiche.misc import format_dict, to_engstr
from pystiche.enc import Encoder

__all__ = [
    "Operator",
    "ComparisonOperator",
    "RegularizationOperator",
    "EncodingOperator",
    "PixelOperator",
    "EncodingComparisonOperator",
    "EncodingRegularizationOperator",
    "PixelComparisonOperator",
    "PixelRegularizationOperator",
    "GuidanceOperator",
    "ComparisonGuidanceOperator",
    "RegularizationGuidanceOperator",
    "EncodingGuidanceOperator",
    "PixelGuidanceOperator",
    "GuidedEncodingComparisonOperator",
    "GuidedEncodingRegularizationOperator",
    "GuidedPixelComparisonOperator",
    "GuidedPixelRegularizationOperator",
]


class Operator(pystiche.object):
    def __init__(self, name: str, score_weight: Numeric = 1.0):
        super().__init__()
        self.name = name
        self.score_weight = score_weight

    def __call__(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.forward(input_image) * self.score_weight

    def extra_str(self) -> str:
        dct = self._descriptions()
        dct.update(self.extra_descriptions())
        return format_dict(dct)

    def _descriptions(self) -> Dict[str, str]:
        dct = OrderedDict()
        dct["Name"] = self.name
        dct["Score weight"] = to_engstr(self.score_weight)
        return dct

    def extra_descriptions(self) -> Dict[str, str]:
        return OrderedDict()

    @abstractmethod
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        pass


class ComparisonOperator(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_image = None
        self._target_repr = None
        self._ctx = None

    def set_target(self, image: torch.Tensor):
        with torch.no_grad():
            repr, ctx = self._process_target(image)
        self.target_image = image.detach()
        self._target_repr = repr.detach()
        if self._is_special_obj(ctx):
            ctx = ctx.detach()
        self._ctx = ctx

    @property
    def has_target_image(self) -> bool:
        return self.target_image is not None

    @abstractmethod
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _process_input(self, image: torch.Tensor, ctx: Any) -> Any:
        pass

    @abstractmethod
    def _process_target(self, image: torch.Tensor) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def _calculate_score(
        self, input_repr: Any, target_repr: Any, ctx: Any
    ) -> torch.Tensor:
        pass


class RegularizationOperator(Operator):
    @abstractmethod
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _process_input(self, image: torch.Tensor) -> Any:
        pass

    @abstractmethod
    def _calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        pass


class EncodingOperator(Operator):
    def __init__(
        self,
        encoder: Encoder,
        layers: Sequence[str],
        name: str,
        layer_weights: Union[str, Sequence[Numeric]] = "mean",
        **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
        encoder.verify_layers(layers)
        self.encoder = encoder
        self.layers = layers
        self.layer_weights = self._verify_layer_weights(layer_weights)

    def _verify_layer_weights(self, layer_weights):
        if isinstance(layer_weights, str):
            num_layers = len(self.layers)
            if layer_weights == "mean":
                return [1.0 / num_layers] * num_layers
            elif layer_weights == "sum":
                return [1.0] * num_layers
        elif isinstance(layer_weights, Sequence):
            assert len(layer_weights) == len(self.layers)
            return layer_weights
        # FIXME: add error message
        raise ValueError

    def _descriptions(self) -> Dict[str, str]:
        dct = super()._descriptions()
        if len(self.layers) == 1:
            dct["Layer"] = self.layers[0]
            weight = self.layer_weights[0]
            if weight != 1.0:
                dct["Layer weight"] = to_engstr(weight)
        else:
            dct["Layers"] = ", ".join(self.layers)
            dct["Layer weights"] = ", ".join(
                [to_engstr(weight) for weight in self.layer_weights]
            )
        return dct

    @abstractmethod
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        pass


class PixelOperator(Operator):
    @abstractmethod
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        pass


class EncodingComparisonOperator(EncodingOperator, ComparisonOperator):
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        # FIXME: do sanity checks if target was set
        target_reprs, ctxs = self._target_repr, self._ctx
        input_reprs = self._process_input(input_image, ctxs)

        layer_scores = [
            self._calculate_score(input_repr, target_repr, ctx)
            for input_repr, target_repr, ctx in zip(input_reprs, target_reprs, ctxs)
        ]
        score = sum(
            [weight * score for weight, score in zip(self.layer_weights, layer_scores)]
        )
        return score

    def _process_input(self, image: torch.Tensor, ctxs: Sequence[Any]) -> Sequence[Any]:
        encs = self.encoder(image, self.layers)
        return self._input_encs_to_reprs(encs, ctxs)

    def _input_encs_to_reprs(self, encs: Sequence[torch.Tensor], ctxs: Sequence[Any]):
        reprs = [self._input_enc_to_repr(enc, ctx) for enc, ctx in zip(encs, ctxs)]
        return pystiche.tuple(reprs)

    @abstractmethod
    def _input_enc_to_repr(self, enc: torch.Tensor, ctx: Any) -> Any:
        pass

    def _process_target(
        self, image: torch.Tensor
    ) -> Tuple[Sequence[Any], Sequence[Any]]:
        encs = self.encoder(image, self.layers)
        return self._target_encs_to_reprs(encs)

    def _target_encs_to_reprs(
        self, encs: Sequence[torch.Tensor]
    ) -> Tuple[Sequence[Any], Sequence[Any]]:
        reprs, ctxs = zip(*[self._target_enc_to_repr(enc) for enc in encs])
        return pystiche.tuple(reprs), pystiche.tuple(ctxs)

    @abstractmethod
    def _target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def _calculate_score(
        self, input_repr: Any, target_repr: Any, ctx: Any
    ) -> torch.Tensor:
        pass


class EncodingRegularizationOperator(EncodingOperator, RegularizationOperator):
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        input_reprs = self._process_input(input_image)

        layer_scores = [self._calculate_score(input_repr) for input_repr in input_reprs]
        score = sum(
            [weight * score for weight, score in zip(self.layer_weights, layer_scores)]
        )
        return score

    def _process_input(self, image: torch.Tensor) -> Sequence[Any]:
        encs = self.encoder(image, self.layers)
        reprs = self._input_encs_to_reprs(encs)
        return reprs

    def _input_encs_to_reprs(self, encs: torch.Tensor):
        reprs = [self.input_enc_to_repr(enc) for enc in encs]
        return pystiche.tuple(reprs)

    @abstractmethod
    def _input_enc_to_repr(self, enc: torch.Tensor) -> Any:
        pass

    @abstractmethod
    def _calculate_score(self, input_repr: Any) -> torch.Tensor:
        pass


class PixelComparisonOperator(PixelOperator, ComparisonOperator):
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        target_repr, ctx = self._target_repr, self._ctx
        input_repr = self._process_input(input_image, ctx)

        score = self._calculate_score(input_repr, target_repr, ctx)
        return score

    def _process_input(self, image: torch.Tensor, ctx: Any) -> Any:
        repr = self._input_image_to_repr(image, ctx)
        return repr

    @abstractmethod
    def _input_image_to_repr(self, image: torch.Tensor, ctx: Any) -> Any:
        pass

    def _process_target(self, image: torch.Tensor) -> Any:
        return self._target_image_to_repr(image)

    @abstractmethod
    def _target_image_to_repr(self, image: torch.Tensor) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def _calculate_score(
        self, input_repr: Any, target_repr: Any, ctx: Any
    ) -> torch.Tensor:
        pass


class PixelRegularizationOperator(PixelOperator, RegularizationOperator):
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        input_repr = self._process_input(input_image)

        score = self._calculate_score(input_repr)
        return score

    def _process_input(self, image: torch.Tensor) -> Any:
        repr = self._input_image_to_repr(image)
        return repr

    @abstractmethod
    def _input_image_to_repr(self, image: torch.Tensor) -> Any:
        pass

    @abstractmethod
    def _calculate_score(self, input_repr: Any) -> torch.Tensor:
        pass


class GuidanceOperator(Operator):
    def __init__(self, name: str, method: str = "simple", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.input_guide = None
        self.method = method

    def set_input_guide(self, guide: torch.Tensor):
        self.input_guide = guide.detach()

    def _apply_guide(self, image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        return image * guide

    @property
    def has_input_guide(self) -> bool:
        return self.input_guide is not None


class ComparisonGuidanceOperator(GuidanceOperator, ComparisonOperator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_guide = None

    def set_target_guide(self, guide: torch.Tensor):
        self.target_guide = guide.detach()

    @property
    def has_target_guide(self) -> bool:
        return self.target_guide is not None


class RegularizationGuidanceOperator(GuidanceOperator, RegularizationOperator):
    pass


class EncodingGuidanceOperator(GuidanceOperator, EncodingOperator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._input_enc_guides = None

    def set_input_guide(self, guide: torch.Tensor):
        super().set_input_guide(guide)
        with torch.no_grad():
            guides = self._calculate_enc_guides(guide)
        self._input_enc_guides = guides.detach()

    def _calculate_enc_guides(self, guide: torch.Tensor):
        guides = self.encoder.propagate_guide(guide, self.layers, self.method)
        return pystiche.tuple(guides)

    def _apply_enc_guides(
        self, encs: Sequence[torch.Tensor], guides: Sequence[torch.Tensor]
    ):
        return pystiche.tuple([enc * guide for enc, guide in zip(encs, guides)])

    @property
    def has_input_enc_guides(self) -> bool:
        return self._input_enc_guides is not None


class PixelGuidanceOperator(GuidanceOperator, PixelOperator):
    pass


class GuidedEncodingComparisonOperator(
    EncodingGuidanceOperator, ComparisonGuidanceOperator, EncodingComparisonOperator
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._target_enc_guides = None

    def set_target_guide(self, guide: torch.Tensor):
        super().set_target_guide(guide)
        with torch.no_grad():
            guides = self._calculate_enc_guides(guide)
        self._target_enc_guides = guides.detach()

    def _input_encs_to_reprs(
        self, encs: Sequence[torch.Tensor], ctxs: Sequence[Any]
    ) -> Sequence[Any]:
        if self.has_input_enc_guides:
            encs = self._apply_enc_guides(encs, self._input_enc_guides)
        return super()._input_encs_to_reprs(encs, ctxs)

    def _target_encs_to_reprs(
        self, encs: Sequence[torch.Tensor]
    ) -> Tuple[Sequence[Any], Sequence[Any]]:
        if self.has_target_enc_guides:
            encs = self._apply_enc_guides(encs, self._target_enc_guides)
        return super()._target_encs_to_reprs(encs)

    @property
    def has_target_enc_guides(self) -> bool:
        return self._target_enc_guides is not None


class GuidedPixelComparisonOperator(
    PixelGuidanceOperator, ComparisonGuidanceOperator, PixelComparisonOperator
):
    def _input_image_to_repr(self, image: torch.Tensor, ctx: Any) -> Any:
        if self.has_input_guide:
            image = self._apply_guide(image, self.input_guide)
        return super()._input_image_to_repr(image, ctx)

    def _target_image_to_repr(self, image: torch.Tensor) -> Tuple[Any, Any]:
        if self.has_target_guide:
            image = self._apply_guide(image, self.target_guide)
        return super()._target_image_to_repr(image)


class GuidedEncodingRegularizationOperator(
    EncodingGuidanceOperator,
    RegularizationGuidanceOperator,
    EncodingRegularizationOperator,
):
    def _input_encs_to_reprs(self, encs: Sequence[torch.Tensor]) -> Sequence[Any]:
        if self.has_input_enc_guides:
            encs = self._apply_enc_guides(encs, self._input_enc_guides)
        return super()._input_encs_to_reprs(encs)


class GuidedPixelRegularizationOperator(
    PixelGuidanceOperator, RegularizationGuidanceOperator, PixelRegularizationOperator
):
    def _input_image_to_repr(self, image: torch.Tensor) -> Any:
        if self.has_input_guide:
            image = self._apply_guide(image, self.input_guide)
        return super()._input_image_to_repr(image)
