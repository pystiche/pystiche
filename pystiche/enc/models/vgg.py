import re
import warnings
from typing import Any, Dict, List, Tuple

import torchvision
from torch import nn
from torch.utils import model_zoo

from pystiche.misc import build_deprecation_message

from ..multi_layer_encoder import MultiLayerEncoder, SingleLayerEncoder
from ..preprocessing import get_preprocessor

MODELS = {
    name: vgg_net
    for name, vgg_net in torchvision.models.vgg.__dict__.items()
    if name.startswith("vgg") and callable(vgg_net)
}

MODEL_URLS = {
    ("torch", arch): url for arch, url in torchvision.models.vgg.model_urls.items()
}
# The caffe weights were created by Karen Simonyan and Andrew Zisserman. See
# https://download.pystiche.org/models/LICENSE for details.
MODEL_URLS.update(
    {
        ("caffe", "vgg16",): "https://download.pystiche.org/models/vgg16-781be684.pth",
        ("caffe", "vgg19",): "https://download.pystiche.org/models/vgg19-74e45263.pth",
    }
)

__all__ = [
    "vgg11_multi_layer_encoder",
    "vgg11_bn_multi_layer_encoder",
    "vgg13_multi_layer_encoder",
    "vgg13_bn_multi_layer_encoder",
    "vgg16_multi_layer_encoder",
    "vgg16_bn_multi_layer_encoder",
    "vgg19_multi_layer_encoder",
    "vgg19_bn_multi_layer_encoder",
]

DEPRECATED_LAYER_PATTERN = re.compile(
    r"^(?P<type>(conv|bn|relu|pool))_(?P<block>\d)(_(?P<depth>\d))?$"
)


class MultiLayerVGGEncoder(MultiLayerEncoder):
    def __init__(
        self, arch: str, weights: str, internal_preprocessing: bool, allow_inplace: bool
    ) -> None:
        self.arch = arch
        self.weights = weights
        self.internal_preprocessing = internal_preprocessing
        self.allow_inplace = allow_inplace

        super().__init__(self._collect_modules())

    def _collect_modules(self) -> List[Tuple[str, nn.Module]]:
        base_model = MODELS[self.arch]()
        url = MODEL_URLS[(self.weights, self.arch)]
        state_dict = model_zoo.load_url(url)
        base_model.load_state_dict(state_dict)
        model = base_model.features

        modules = []
        if self.internal_preprocessing:
            modules.append(("preprocessing", get_preprocessor(self.weights)))

        block = depth = 1
        for module in model.children():
            if isinstance(module, nn.Conv2d):
                name = f"conv{block}_{depth}"
            elif isinstance(module, nn.BatchNorm2d):
                name = f"bn{block}_{depth}"
            elif isinstance(module, nn.ReLU):
                if not self.allow_inplace:
                    module = nn.ReLU(inplace=False)
                name = f"relu{block}_{depth}"
                # each ReLU layer increases the depth of the current block
                depth += 1
            else:  # isinstance(module, nn.MaxPool2d):
                name = f"pool{block}"
                # each pooling layer marks the end of the current block
                block += 1
                depth = 1

            modules.append((name, module))

        return modules

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["arch"] = self.arch
        dct["weights"] = self.weights
        if not self.internal_preprocessing:
            dct["internal_preprocessing"] = self.internal_preprocessing
        if self.allow_inplace:
            dct["allow_inplace"] = self.allow_inplace
        return dct

    def extract_encoder(self, layer: str) -> SingleLayerEncoder:
        match = DEPRECATED_LAYER_PATTERN.match(layer)
        if match is not None:
            old_layer = layer
            type = match.group("type")
            block = match.group("block")
            depth = match.group("depth")
            layer = f"{type}{block}"
            if depth is not None:
                layer += f"_{depth}"
            msg = build_deprecation_message(
                f"The layer pattern {old_layer}",
                "0.4.0",
                info=(
                    f"Please remove the underscore between the layer type and the "
                    f"block number, i.e. {layer}."
                ),
                url="https://github.com/pmeier/pystiche/issues/125",
            )
            warnings.warn(msg)
        return super().extract_encoder(layer)


class VGGEncoder(MultiLayerVGGEncoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        msg = build_deprecation_message(
            "The class VGGEncoder",
            "0.4.0",
            info="It was replaced by MultiLayerVGGEncoder.",
        )
        warnings.warn(msg)
        super().__init__(*args, **kwargs)


def _vgg_encoder_multi_layer_encoder(
    arch: str,
    weights: str = "torch",
    preprocessing: bool = True,
    allow_inplace: bool = False,
) -> MultiLayerVGGEncoder:
    return MultiLayerVGGEncoder(arch, weights, preprocessing, allow_inplace)


def vgg11_multi_layer_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg11", **kwargs)


def vgg11_bn_multi_layer_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg11_bn", **kwargs)


def vgg13_multi_layer_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg13", **kwargs)


def vgg13_bn_multi_layer_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg13_bn", **kwargs)


def vgg16_multi_layer_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg16", **kwargs)


def vgg16_bn_multi_layer_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg16_bn", **kwargs)


def vgg19_multi_layer_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg19", **kwargs)


def vgg19_bn_multi_layer_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg19_bn", **kwargs)


def _vgg_encoder(arch: str, **kwargs: Any) -> MultiLayerVGGEncoder:
    msg = build_deprecation_message(
        f"The function {arch}_encoder",
        "0.4.0",
        info=f"It was replaced by {arch}_multi_layer_encoder.",
    )
    warnings.warn(msg)
    return MultiLayerVGGEncoder(arch, **kwargs)


def vgg11_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg11", **kwargs)


def vgg11_bn_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg11_bn", **kwargs)


def vgg13_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg13", **kwargs)


def vgg13_bn_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg13_bn", **kwargs)


def vgg16_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg16", **kwargs)


def vgg16_bn_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg16_bn", **kwargs)


def vgg19_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg19", **kwargs)


def vgg19_bn_encoder(**kwargs: Any) -> MultiLayerVGGEncoder:
    return _vgg_encoder_multi_layer_encoder("vgg19_bn", **kwargs)
