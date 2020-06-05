from torch import nn, optim
from torch.optim.optimizer import Optimizer

from pystiche.enc import MultiLayerEncoder, vgg16_multi_layer_encoder
from pystiche.image.transforms import CaffePostprocessing, CaffePreprocessing

__all__ = [
    "johnson_alahi_li_2016_preprocessor",
    "johnson_alahi_li_2016_postprocessor",
    "johnson_alahi_li_2016_multi_layer_encoder",
    "johnson_alahi_li_2016_optimizer",
]


def johnson_alahi_li_2016_preprocessor() -> CaffePreprocessing:
    return CaffePreprocessing()


def johnson_alahi_li_2016_postprocessor() -> CaffePostprocessing:
    return CaffePostprocessing()


def johnson_alahi_li_2016_multi_layer_encoder(impl_params=True) -> MultiLayerEncoder:
    return vgg16_multi_layer_encoder(
        weights="caffe", preprocessing=not impl_params, allow_inplace=True
    )


def johnson_alahi_li_2016_optimizer(transformer: nn.Module) -> Optimizer:
    return optim.Adam(transformer.parameters(), lr=1e-3)
