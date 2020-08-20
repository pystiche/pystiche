import warnings
from typing import Any, Dict, List, Optional, Tuple

from torch import nn
from torchvision.models import alexnet
from torchvision.models.alexnet import model_urls as TORCH_MODEL_URLS

from pystiche.misc import build_deprecation_message

from .utils import ModelMultiLayerEncoder, select_url

__all__ = ["AlexNetMultiLayerEncoder", "alexnet_multi_layer_encoder"]


MODEL_URLS = {"torch": TORCH_MODEL_URLS["alexnet"]}


def _make_description() -> str:
    return r"""Multi-layer encoder based on :class:`~torchvision.models.AlexNet`.

    The :class:`~torchvision.models.AlexNet` architecture was introduced by
    Krizhevsky, Sutskever, and Hinton in :cite:`KSH2012`.
    """


def _make_docstring(body: str) -> str:
    return f"{_make_description()}\n{body}"


class AlexNetMultiLayerEncoder(ModelMultiLayerEncoder):
    __doc__ = _make_docstring(
        r"""    Args:
        pretrained: If ``True``, loads builtin weights. Defaults to ``True``.
        framework: Name of the framework that was used to train the builtin weights.
            Defaults to ``"torch"``.
        kwargs: Optional arguments of :class:`ModelMultiLayerEncoder` .

    Raises:
        RuntimeError: If ``pretrained`` and no weights are available for the
        ``framework``.
        """
    )

    def __init__(self, weights: Optional[str] = None, **kwargs: Any) -> None:
        if weights is not None:
            msg = build_deprecation_message(
                "The parameter weights", "0.6.0", info="It was renamed to framework"
            )
            warnings.warn(msg, UserWarning)
            kwargs["framework"] = weights

        super().__init__(**kwargs)

    def state_dict_url(self, framework: str) -> str:
        return select_url(MODEL_URLS, framework)

    def collect_modules(
        self, inplace: bool
    ) -> Tuple[List[Tuple[str, nn.Module]], Dict[str, str]]:
        model = alexnet(pretrained=False)

        modules = []
        state_dict_key_map = {}
        block = 1
        for idx, module in model.features.named_children():
            if isinstance(module, nn.Conv2d):
                name = f"conv{block}"
            elif isinstance(module, nn.ReLU):
                module = nn.ReLU(inplace=inplace)
                name = f"relu{block}"
                if block in (3, 4):
                    # in the third and forth block the ReLU layer marks the end of the
                    # block
                    block += 1
            else:  # isinstance(module, nn.MaxPool2d):
                name = f"pool{block}"
                # each pooling layer marks the end of the current block
                block += 1

            modules.append((name, module))
            state_dict_key_map.update(
                {
                    f"features.{idx}.{key}": f"{name}.{key}"
                    for key in module.state_dict().keys()
                }
            )

        return modules, state_dict_key_map


def alexnet_multi_layer_encoder(**kwargs: Any) -> AlexNetMultiLayerEncoder:
    return AlexNetMultiLayerEncoder(**kwargs)


alexnet_multi_layer_encoder.__doc__ = _make_docstring(
    r"""    Args:
        kwargs: Optional arguments of :class:`AlexNetMultiLayerEncoder` .
"""
)
