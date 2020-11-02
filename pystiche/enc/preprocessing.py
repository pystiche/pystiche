from torch import nn

from pystiche.image.transforms import CaffePreprocessing, TorchPreprocessing
from pystiche.misc import suppress_warnings

PREPROCESSORS = {"torch": TorchPreprocessing, "caffe": CaffePreprocessing}

__all__ = ["get_preprocessor"]


def get_preprocessor(framework: str) -> nn.Module:
    with suppress_warnings():
        return PREPROCESSORS[framework]()
