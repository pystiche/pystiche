from typing import Optional
from collections import OrderedDict
from torch.utils import model_zoo
from torch import nn
import torchvision
from pystiche.image.transforms import TorchPreprocessing, CaffePreprocessing
from .encoder import Encoder

VGG_NETS = {
    name: vgg_net
    for name, vgg_net in torchvision.models.vgg.__dict__.items()
    if name.startswith("vgg") and callable(vgg_net)
}

MODEL_URLS = {
    ("torch", name): url for name, url in torchvision.models.vgg.model_urls.items()
}
MODEL_URLS.update(
    {
        (
            "caffe",
            "vgg16",
        ): "https://store.th-owl.de:443/ssf/s/readFile/share/5899/2114173033433010752/publicLink/vgg16-781be684.pth",
        (
            "caffe",
            "vgg19",
        ): "https://store.th-owl.de:443/ssf/s/readFile/share/5901/-7191405041549013253/publicLink/vgg19-74e45263.pth",
    }
)

PREPROCESSORS = {"torch": TorchPreprocessing(), "caffe": CaffePreprocessing()}

__all__ = [
    "vgg11_encoder",
    "vgg11_bn_encoder",
    "vgg13_encoder",
    "vgg13_bn_encoder",
    "vgg16_encoder",
    "vgg16_bn_encoder",
    "vgg19_encoder",
    "vgg19_bn_encoder",
]


class VGGEncoder(Encoder):
    def __init__(self, vgg_net: nn.Module, preprocessor: Optional[nn.Module] = None):
        modules = OrderedDict()
        if preprocessor is not None:
            modules["preprocessing"] = preprocessor
        stack = depth = 1
        for module in vgg_net.features.children():
            if isinstance(module, nn.Conv2d):
                name = "conv_{0}_{1}".format(stack, depth)
            elif isinstance(module, nn.BatchNorm2d):
                name = "bn_{0}_{1}".format(stack, depth)
            elif isinstance(module, nn.ReLU):
                module = nn.ReLU(inplace=False)
                name = "relu_{0}_{1}".format(stack, depth)
                # each ReLU layer increases the depth of the current stack
                depth += 1
            else:  # isinstance(module, nn.MaxPool2d):
                name = "pool_{0}".format(stack)
                # each pooling layer marks the end of the current stack
                stack += 1
                depth = 1

            modules[name] = module

        super().__init__(modules)


def _vgg_encoder(
    arch: str, weights: str = "torch", preprocessing: bool = True
) -> VGGEncoder:
    vgg_net = VGG_NETS[arch](init_weights=False)
    url = MODEL_URLS[(weights, arch)]
    vgg_net.load_state_dict(model_zoo.load_url(url))
    preprocessor = PREPROCESSORS[weights] if preprocessing else None
    return VGGEncoder(vgg_net, preprocessor)


# TODO: do these need annotating?
def vgg11_encoder(**kwargs):
    return _vgg_encoder("vgg11", **kwargs)


def vgg11_bn_encoder(**kwargs):
    return _vgg_encoder("vgg11_bn", **kwargs)


def vgg13_encoder(**kwargs):
    return _vgg_encoder("vgg13", **kwargs)


def vgg13_bn_encoder(**kwargs):
    return _vgg_encoder("vgg13_bn", **kwargs)


def vgg16_encoder(**kwargs):
    return _vgg_encoder("vgg16", **kwargs)


def vgg16_bn_encoder(**kwargs):
    return _vgg_encoder("vgg16_bn", **kwargs)


def vgg19_encoder(**kwargs):
    return _vgg_encoder("vgg19", **kwargs)


def vgg19_bn_encoder(**kwargs):
    return _vgg_encoder("vgg19_bn", **kwargs)
