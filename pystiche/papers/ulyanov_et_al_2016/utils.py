from torch import nn, optim
from typing import Optional, Any, List
from torch.optim.lr_scheduler import ExponentialLR
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


class DelayedExponentialLR(ExponentialLR):
    def __init__(
        self, optimizer: Optimizer, gamma: float, delay: int, **kwargs: Any
    ) -> None:
        self.delay = delay
        super().__init__(optimizer, gamma, **kwargs)

    def get_lr(self) -> List[float]:
        exp = self.last_epoch - self.delay + 1
        if exp > 0:
            return [base_lr * self.gamma ** exp for base_lr in self.base_lrs]
        else:
            return self.base_lrs


def ulyanov_et_al_2016_lr_scheduler(
    optimizer: Optional[Optimizer] = None, impl_params: bool = True,
) -> Optional[ExponentialLR, None]:
    if optimizer is None:
        return None
    if impl_params:
        lr_scheduler = ExponentialLR(optimizer, 0.8)
    else:
        lr_scheduler = DelayedExponentialLR(optimizer, 0.7, 5)
    return lr_scheduler
