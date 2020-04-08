from typing import Union, Optional, Callable
import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import pystiche
from pystiche.optim import (
    OptimLogger,
    default_transformer_epoch_optim_loop,
)
from ..common_utils import batch_up_image
from .modules import SanakoyeuEtAl2018Encoder, SanakoyeuEtAl2018Decoder, SanakoyeuEtAl2018Discriminator, sanakoyeu_et_al_2018_discriminator, SanakoyeuEtAl2018TransformerBlock


from .utils import sanakoyeu_et_al_2018_optimizer


__all__ = [
    "sanakoyeu_et_al_2018_training",
    "SanakoyeuEtAl2018Encoder",
    "SanakoyeuEtAl2018Decoder",
    "SanakoyeuEtAl2018Discriminator"
]


