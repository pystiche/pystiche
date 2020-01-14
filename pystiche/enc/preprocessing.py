from torch import nn
from pystiche.image.transforms import TorchPreprocessing, CaffePreprocessing

PREPROCESSORS = {"torch": TorchPreprocessing(), "caffe": CaffePreprocessing()}

__all__ = ["get_preprocessor"]


def get_preprocessor(framework: str) -> nn.Module:
    return PREPROCESSORS[framework]
