from typing import Any, Dict, List, Tuple

import torchvision
from torch import nn
from torch.utils import model_zoo
from torchvision.models import VGG

from ..multi_layer_encoder import MultiLayerEncoder
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
    "VGGMultiLayerEncoder",
    "vgg11_multi_layer_encoder",
    "vgg11_bn_multi_layer_encoder",
    "vgg13_multi_layer_encoder",
    "vgg13_bn_multi_layer_encoder",
    "vgg16_multi_layer_encoder",
    "vgg16_bn_multi_layer_encoder",
    "vgg19_multi_layer_encoder",
    "vgg19_bn_multi_layer_encoder",
]


class VGGMultiLayerEncoder(MultiLayerEncoder):
    r"""Multi-layer encoder based on the VGG architecture that was introduced by
    Simonyan and Zisserman in :cite:`SZ2014`.

    Args:
        arch: Specific architecture. Has to match ``"vgg(11|13|16|19)(_bn)?"``.
        weights: Framework to load the pretrained weights from. Can be ``"torch"`` or
            ``"caffe"``. Defaults to ``"torch"``.

            .. note::
                Caffe weights are only available for ``"vgg16"`` and ``"vgg19"``.
        internal_preprocessing: If ``True``, adds a preprocessing layer for the
            selected ``weights`` as first layer. Defaults to ``"True"``.
        allow_inplace: If ``True``, allows inplace operations in the
            :class:`torch.nn.Relu` layers to reduce the memory requirement during the
            forward pass. Defaults to ``False``.

            .. warning::
                After performing an inplace operation in a :class:`torch.nn.Relu` layer
                the encoding of the previous :class:`torch.nn.Conv2d` layer is no
                longer accessible. Only set this to ``True`` if you are sure that you
                do **not** need the encodings of the :class:`torch.nn.Conv2d` layers.
    """

    def __init__(
        self,
        arch: str,
        weights: str = "torch",
        internal_preprocessing: bool = True,
        allow_inplace: bool = False,
    ) -> None:
        self.arch = arch
        self.weights = weights
        self.internal_preprocessing = internal_preprocessing
        self.allow_inplace = allow_inplace

        super().__init__(self._collect_modules())

    def _collect_modules(self) -> List[Tuple[str, nn.Module]]:
        base_model = MODELS[self.arch](pretrained=False)
        self._load_weights(base_model)

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

    def _load_weights(self, base_model: VGG) -> None:
        url = MODEL_URLS[(self.weights, self.arch)]
        state_dict = model_zoo.load_url(url)
        base_model.load_state_dict(state_dict)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["arch"] = self.arch
        dct["weights"] = self.weights
        if not self.internal_preprocessing:
            dct["internal_preprocessing"] = self.internal_preprocessing
        if self.allow_inplace:
            dct["allow_inplace"] = self.allow_inplace
        return dct


def vgg11_multi_layer_encoder(**kwargs: Any) -> VGGMultiLayerEncoder:
    r"""Multi-layer encoder based on the VGG11 architecture.

    Args:
        **kwargs: Optional parameters for :class:`VGGMultiLayerEncoder`.
    """
    return VGGMultiLayerEncoder("vgg11", **kwargs)


def vgg11_bn_multi_layer_encoder(**kwargs: Any) -> VGGMultiLayerEncoder:
    r"""Multi-layer encoder based on the VGG11 architecture with
    :class:`torch.nn.BatchNorm2d` layers.

    Args:
        **kwargs: Optional parameters for :class:`VGGMultiLayerEncoder`.
    """
    return VGGMultiLayerEncoder("vgg11_bn", **kwargs)


def vgg13_multi_layer_encoder(**kwargs: Any) -> VGGMultiLayerEncoder:
    r"""Multi-layer encoder based on the VGG13 architecture.

    Args:
        **kwargs: Optional parameters for :class:`VGGMultiLayerEncoder`.
    """
    return VGGMultiLayerEncoder("vgg13", **kwargs)


def vgg13_bn_multi_layer_encoder(**kwargs: Any) -> VGGMultiLayerEncoder:
    r"""Multi-layer encoder based on the VGG13 architecture with
    :class:`torch.nn.BatchNorm2d` layers.

    Args:
        **kwargs: Optional parameters for :class:`VGGMultiLayerEncoder`.
    """
    return VGGMultiLayerEncoder("vgg13_bn", **kwargs)


def vgg16_multi_layer_encoder(**kwargs: Any) -> VGGMultiLayerEncoder:
    r"""Multi-layer encoder based on the VGG16 architecture.

    Args:
        **kwargs: Optional parameters for :class:`VGGMultiLayerEncoder`.
    """
    return VGGMultiLayerEncoder("vgg16", **kwargs)


def vgg16_bn_multi_layer_encoder(**kwargs: Any) -> VGGMultiLayerEncoder:
    r"""Multi-layer encoder based on the VGG16 architecture with
    :class:`torch.nn.BatchNorm2d` layers.

    Args:
        **kwargs: Optional parameters for :class:`VGGMultiLayerEncoder`.
    """
    return VGGMultiLayerEncoder("vgg16_bn", **kwargs)


def vgg19_multi_layer_encoder(**kwargs: Any) -> VGGMultiLayerEncoder:
    r"""Multi-layer encoder based on the VGG19 architecture.

    Args:
        **kwargs: Optional parameters for :class:`VGGMultiLayerEncoder`.
    """
    return VGGMultiLayerEncoder("vgg19", **kwargs)


def vgg19_bn_multi_layer_encoder(**kwargs: Any) -> VGGMultiLayerEncoder:
    r"""Multi-layer encoder based on the VGG19 architecture with
    :class:`torch.nn.BatchNorm2d` layers.

    Args:
        **kwargs: Optional parameters for :class:`VGGMultiLayerEncoder`.
    """
    return VGGMultiLayerEncoder("vgg19_bn", **kwargs)
