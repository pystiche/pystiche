import functools
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torchvision
from torch import hub, nn

from pystiche.misc import build_deprecation_message

from .utils import ModelMultiLayerEncoder
from .utils import select_url as _select_url

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

ARCH_PATTERN = re.compile(r"^vgg(?P<num_layers>(11|13|16|19)+)(?P<batch_norm>_bn)?$")

NUM_LAYERS_TO_CONFIGURATION = {
    11: "A",
    13: "B",
    16: "D",
    19: "E",
}


def _parse_arch(arch: str) -> Tuple[int, bool]:
    match = ARCH_PATTERN.match(arch)
    if match is None:
        raise ValueError(
            f"Unknown arch '{arch}'. It has to match 'vgg_(11|13|16|19)(_bn)?'"
        )

    num_layers = int(match.group("num_layers"))
    batch_norm = match.group("batch_norm") is not None

    return num_layers, batch_norm


def _make_description(arch: str, multi_layer_encoder: bool) -> str:
    num_layers, batch_norm = _parse_arch(arch)

    if multi_layer_encoder:
        short = (
            f"Multi-layer encoder based on :class:`~torchvision.models.VGG` "
            f"{num_layers}{' with batch normalization' if batch_norm else ''}."
        )
    else:
        short = f":class:`~torchvision.models.VGG` {num_layers} model"

    long = (
        f"    The :class:`~torchvision.models.VGG` architecture was introduced by "
        f"Krizhevsky, Sutskever, and Hinton in :cite:`KSH2012`. VGG{num_layers} "
        f"corresponds to configuration ``{NUM_LAYERS_TO_CONFIGURATION[num_layers]}`` "
        f"in the paper."
    )
    return "\n".join((short, "", long))


def _make_vgg_docstring(arch: str) -> str:
    description = _make_description(arch, multi_layer_encoder=False)
    args = r"""Args:
    pretrained: If ``True``, loads weights from training on
        :class:`~torchvision.models.ImageNet`. Defaults to ``False``.
    framework: Framework that was used to train the model. Can be one of
        ``"torch"`` (default) or ``"caffe"``.
        .. note::

            The weights for ``"caffe"`` were generated by Karen Simonyan and
            Andrew Zisserman. See https://download.pystiche.org/models/LICENSE for
            details.
    progress: If ``True``, displays a progress bar to STDERR during download of
        pretrained weights. Defaults to ``True``.
    num_classes: Size of the output layer. Defaults to ``1000``.
        .. note::

            Pretrained weights are only available for ``num_classes == 1000``.
    """
    return "\n".join((description, "", args))


def select_url(arch: str, framework: str) -> str:
    def format(key: Tuple[str, str]) -> str:
        arch, framework = key
        return "\n".join((f"arch={arch}", f"framework={framework}"))

    return _select_url(MODEL_URLS, (arch, framework), format=format)


def _vgg_loader(arch: str) -> Callable[..., torchvision.models.VGG]:
    loader = cast(
        Callable[..., torchvision.models.VGG], getattr(torchvision.models, arch)
    )

    def vgg(
        pretrained: bool = False,
        framework: str = "torch",
        progress: bool = True,
        num_classes: int = 1000,
    ) -> torchvision.models.VGG:
        if pretrained and num_classes != 1000:
            raise RuntimeError

        model = loader(pretrained=False, num_classes=num_classes)

        if not pretrained:
            return model

        state_dict = hub.load_state_dict_from_url(
            select_url(arch, framework), progress=progress, check_hash=True,
        )
        model.load_state_dict(state_dict)
        return model

    vgg.__doc__ = _make_vgg_docstring(arch)

    return vgg


TORCH_MODEL_URLS = torchvision.models.vgg.model_urls
ARCHS = tuple(TORCH_MODEL_URLS.keys())
MODEL_URLS = {(arch, "torch"): TORCH_MODEL_URLS[arch] for arch in ARCHS}
MODEL_URLS.update(
    {
        ("vgg16", "caffe"): "https://download.pystiche.org/models/vgg16-781be684.pth",
        ("vgg19", "caffe"): "https://download.pystiche.org/models/vgg19-74e45263.pth",
    }
)
MODELS = {arch: _vgg_loader(arch) for arch in ARCHS}


class VGGMultiLayerEncoder(ModelMultiLayerEncoder):
    r"""Multi-layer encoder based on :class:`~torchvision.models.VGG`.

    The :class:`~torchvision.models.VGG` architecture was introduced by  Krizhevsky,
    Sutskever, and Hinton in :cite:`KSH2012`

    Args:
        arch: :class:`~torchvision.models.VGG` architecture. Has to match
            ``"vgg(11|13|16|19)(_bn)?"``.
        pretrained: If ``True``, loads builtin weights. Defaults to ``True``.
        framework: Name of the framework that was used to train the builtin weights.
            Defaults to ``"torch"``.
        kwargs: Optional arguments of :class:`ModelMultiLayerEncoder` .

    Raises:
        RuntimeError: If ``pretrained is True`` and no weights are available for the
            combination of ``arch`` and ``framework``.
    """

    def __init__(self, arch: str, weights: Optional[str] = None, **kwargs: Any) -> None:
        if weights is not None:
            msg = build_deprecation_message(
                "The parameter weights", "0.6.0", info="It was renamed to framework"
            )
            warnings.warn(msg, UserWarning)
            kwargs["framework"] = weights

        _parse_arch(arch)
        self.arch = arch
        super().__init__(**kwargs)

    def state_dict_url(self, framework: str) -> str:
        return select_url(self.arch, framework)

    def collect_modules(
        self, inplace: bool
    ) -> Tuple[List[Tuple[str, nn.Module]], Dict[str, str]]:
        model = MODELS[self.arch](pretrained=False)

        modules = []
        state_dict_key_map = {}
        block = depth = 1
        for idx, module in model.features.named_children():
            if isinstance(module, nn.Conv2d):
                name = f"conv{block}_{depth}"
            elif isinstance(module, nn.BatchNorm2d):
                name = f"bn{block}_{depth}"
            elif isinstance(module, nn.ReLU):
                module = nn.ReLU(inplace=inplace)
                name = f"relu{block}_{depth}"
                # each ReLU layer increases the depth of the current block
                depth += 1
            else:  # isinstance(module, nn.MaxPool2d):
                name = f"pool{block}"
                # each pooling layer marks the end of the current block
                block += 1
                depth = 1

            modules.append((name, module))
            state_dict_key_map.update(
                {
                    f"features.{idx}.{key}": f"{name}.{key}"
                    for key in module.state_dict().keys()
                }
            )

        return modules, state_dict_key_map

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()

        dct["arch"] = self.arch
        dct.move_to_end("arch", last=False)  # type: ignore[attr-defined]

        return dct


def _make_vgg_multi_layer_encoder_docstring(arch: str) -> str:
    description = _make_description(arch, multi_layer_encoder=True)
    args = r"""    Args:
        kwargs: Optional arguments of :class:`VGGMultiLayerEncoder` .
    """
    return "\n".join((description, "", args))


def _update_loader_magic(loader: Callable, name: str, doc: str) -> None:
    loader.__module__ = VGGMultiLayerEncoder.__module__
    loader.__name__ = loader.__qualname__ = name
    loader.__annotations__ = {
        param: annotation
        for param, annotation in VGGMultiLayerEncoder.__init__.__annotations__.items()
        if param != "arch"
    }
    loader.__doc__ = doc


for arch in ARCHS:
    name = f"{arch}_multi_layer_encoder"
    doc = _make_vgg_multi_layer_encoder_docstring(arch)
    loader = functools.partial(VGGMultiLayerEncoder, arch)
    _update_loader_magic(loader, name, doc)
    locals()[name] = loader
