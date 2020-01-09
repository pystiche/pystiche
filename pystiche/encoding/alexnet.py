from typing import Optional
from collections import OrderedDict
from torch.utils import model_zoo
from torch import nn
import torchvision
from pystiche.image.transforms import TorchPreprocessing, CaffePreprocessing
from .encoder import Encoder

MODELS = {"alexnet": torchvision.models.alexnet}

MODEL_URLS = {
    ("alexnet", "torch"): "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
}

PREPROCESSORS = {"torch": TorchPreprocessing(), "caffe": CaffePreprocessing()}

__all__ = ["alexnet_encoder"]


class AlexNetEncoder(Encoder):
    def __init__(self, model: nn.Module, preprocessor: Optional[nn.Module] = None):
        modules = OrderedDict()
        if preprocessor is not None:
            modules["preprocessing"] = preprocessor
        block = 1
        for module in model.features.children():
            if isinstance(module, nn.Conv2d):
                name = f"conv_{block}"
            elif isinstance(module, nn.ReLU):
                module = nn.ReLU(inplace=False)
                name = f"relu_{block}"
            else:  # isinstance(module, nn.MaxPool2d):
                name = f"pool_{block}"
                # each pooling layer marks the end of the current block
                block += 1

            modules[name] = module

        super().__init__(modules)


def alexnet_encoder(weights: str = "torch", preprocessing=True):
    model = torchvision.models.alexnet()
    url = MODEL_URLS[("alexnet", weights)]
    model.load_state_dict(model_zoo.load_url(url))
    preprocessor = PREPROCESSORS[weights] if preprocessing else None
    return AlexNetEncoder(model, preprocessor)
