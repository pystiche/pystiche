from torch import optim, nn
from torch.optim.optimizer import Optimizer
from pystiche.image import (
    CaffePreprocessing,
    CaffePostprocessing,
)
from pystiche.enc import MultiLayerEncoder, vgg16_encoder

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
    return vgg16_encoder(
        weights="caffe", preprocessing=not impl_params, allow_inplace=True
    )


def johnson_alahi_li_2016_optimizer(transformer: nn.Module) -> Optimizer:
    return optim.Adam(transformer.parameters(), lr=1e-3)
