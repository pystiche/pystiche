from torch import nn

from pystiche.image import CaffePreprocessing, TorchPreprocessing

PREPROCESSORS = {"torch": TorchPreprocessing, "caffe": CaffePreprocessing}

__all__ = ["get_preprocessor"]


def get_preprocessor(framework: str) -> nn.Module:
    return PREPROCESSORS[framework]()
