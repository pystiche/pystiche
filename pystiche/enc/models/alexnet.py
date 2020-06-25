from typing import Any, Dict, List, Tuple

from torch import nn
from torch.utils import model_zoo
from torchvision.models import AlexNet, alexnet

from ..multi_layer_encoder import MultiLayerEncoder
from ..preprocessing import get_preprocessor

MODEL_URLS = {"torch": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"}


__all__ = ["AlexNetMultiLayerEncoder", "alexnet_multi_layer_encoder"]


class AlexNetMultiLayerEncoder(MultiLayerEncoder):
    r"""Multi-layer encoder based on the AlexNet architecture that was introduced by
    Krizhevsky, Sutskever, and Hinton in :cite:`KSH2012`.

    Args:
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
        weights: str = "torch",
        internal_preprocessing: bool = True,
        allow_inplace: bool = False,
    ) -> None:
        self.weights = weights
        self.internal_preprocessing = internal_preprocessing
        self.allow_inplace = allow_inplace
        super().__init__(self._collect_modules())

    def _collect_modules(self) -> List[Tuple[str, nn.Module]]:
        base_model = alexnet(pretrained=False)
        self._load_weights(base_model)

        model = base_model.features
        modules = []
        if self.internal_preprocessing:
            modules.append(("preprocessing", get_preprocessor(self.weights)))
        block = 1
        for module in model.children():
            if isinstance(module, nn.Conv2d):
                name = f"conv{block}"
            elif isinstance(module, nn.ReLU):
                if not self.allow_inplace:
                    module = nn.ReLU(inplace=False)
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

        return modules

    def _load_weights(self, base_model: AlexNet) -> None:
        url = MODEL_URLS[self.weights]
        state_dict = model_zoo.load_url(url)
        base_model.load_state_dict(state_dict)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["weights"] = self.weights
        if not self.internal_preprocessing:
            dct["internal_preprocessing"] = self.internal_preprocessing
        if self.allow_inplace:
            dct["allow_inplace"] = self.allow_inplace
        return dct


def alexnet_multi_layer_encoder(**kwargs: Any) -> AlexNetMultiLayerEncoder:
    r"""Multi-layer encoder based on the AlexNet architecture.

    Args:
        **kwargs: Optional parameters for :class:`AlexNetMultiLayerEncoder`.
    """
    return AlexNetMultiLayerEncoder(**kwargs)
