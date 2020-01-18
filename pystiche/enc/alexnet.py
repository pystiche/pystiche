from collections import OrderedDict
from torch.utils import model_zoo
from torch import nn
from torchvision.models import alexnet
from .encoder import MultiLayerEncoder
from .preprocessing import get_preprocessor


MODEL_URLS = {"torch": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"}


__all__ = ["alexnet_encoder"]


class AlexNetEncoder(MultiLayerEncoder):
    def __init__(self, weights: str, preprocessing: bool, allow_inplace: bool):
        self.weights = weights
        self.preprocessing = preprocessing
        self.allow_inplace = allow_inplace

        super().__init__(self._collect_modules())

    def collect_modules(self):
        base_model = alexnet()
        url = MODEL_URLS[self.weights]
        state_dict = model_zoo.load_url(url)
        base_model.load_state_dict(state_dict)
        model = base_model.features

        modules = OrderedDict()
        if self.preprocessing:
            modules["preprocessing"] = get_preprocessor(self.weights)
        block = 1
        for module in model.children():
            if isinstance(module, nn.Conv2d):
                name = f"conv_{block}"
            elif isinstance(module, nn.ReLU):
                if not self.allow_inplace:
                    module = nn.ReLU(inplace=False)
                name = f"relu_{block}"
            else:  # isinstance(module, nn.MaxPool2d):
                name = f"pool_{block}"
                # each pooling layer marks the end of the current block
                block += 1

            modules[name] = module

    def extra_repr(self):
        extras = [f"weights={self.weights}"]
        if not self.preprocessing:
            extras.append(f"preprocessing={self.preprocessing}")
        if self.allow_inplace:
            extras.append(f"allow_inplace={self.allow_inplace}")
        return ",".join(extras)


def alexnet_encoder(
    weights: str = "torch", preprocessing=True, allow_inplace: bool = False
):
    return AlexNetEncoder(weights, preprocessing, allow_inplace)
