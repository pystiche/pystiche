from torch import nn, optim
from torch.optim.optimizer import Optimizer
from pystiche.image import CaffePreprocessing, CaffePostprocessing

from pystiche.enc import MultiLayerEncoder, vgg19_encoder


def ulyanov_et_al_2016_multi_layer_encoder() -> MultiLayerEncoder:
    return vgg19_encoder(weights="caffe", allow_inplace=True)


def ulyanov_et_al_2016_preprocessor() -> CaffePreprocessing:
    return CaffePreprocessing()


def ulyanov_et_al_2016_postprocessor() -> CaffePostprocessing:
    return CaffePostprocessing()


def ulyanov_et_al_2016_optimizer(
    transformer: nn.Module, impl_params: bool = True, instance_norm: bool = True
) -> Optimizer:
    if impl_params:
        lr = 1e-3 if instance_norm else 1e-1
    else:
        lr = 1e-1
    return optim.Adam(transformer.parameters(), lr=lr)
