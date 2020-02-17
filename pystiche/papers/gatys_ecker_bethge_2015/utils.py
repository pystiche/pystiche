import torch
from torch import optim, nn
from torch.optim.optimizer import Optimizer
import pystiche
from pystiche.image import CaffePreprocessing, CaffePostprocessing
from pystiche.enc import MultiLayerEncoder, vgg19_encoder

__all__ = [
    "gatys_ecker_bethge_2015_preprocessor",
    "gatys_ecker_bethge_2015_postprocessor",
    "gatys_ecker_bethge_2015_optimizer",
    "gatys_ecker_bethge_2015_multi_layer_encoder",
]


def gatys_ecker_bethge_2015_preprocessor() -> nn.Module:
    return CaffePreprocessing()


def gatys_ecker_bethge_2015_postprocessor() -> nn.Module:
    return CaffePostprocessing()


def gatys_ecker_bethge_2015_multi_layer_encoder(
    impl_params: bool = True,
) -> MultiLayerEncoder:
    multi_layer_encoder = vgg19_encoder(
        weights="caffe", preprocessing=False, allow_inplace=True
    )
    if impl_params:
        return multi_layer_encoder

    for name, module in multi_layer_encoder.named_children():
        if isinstance(module, nn.MaxPool2d):
            multi_layer_encoder._modules[name] = nn.AvgPool2d(
                **pystiche.pool_module_meta(module)
            )
    return multi_layer_encoder


def gatys_ecker_bethge_2015_optimizer(input_image: torch.Tensor) -> Optimizer:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)
