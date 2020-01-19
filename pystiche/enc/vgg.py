from collections import OrderedDict
from torch.utils import model_zoo
from torch import nn
import torchvision
from .encoder import MultiLayerEncoder
from .preprocessing import get_preprocessor

MODELS = {
    name: vgg_net
    for name, vgg_net in torchvision.models.vgg.__dict__.items()
    if name.startswith("vgg") and callable(vgg_net)
}

MODEL_URLS = {
    ("torch", arch): url for arch, url in torchvision.models.vgg.model_urls.items()
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


class VGGEncoder(MultiLayerEncoder):
    def __init__(self, arch: str, weights: str, internal_preprocessing, allow_inplace):
        self.arch = arch
        self.weights = weights
        self.internal_preprocessing = internal_preprocessing
        self.allow_inplace = allow_inplace

        super().__init__(self._collect_modules())

    def _collect_modules(self):
        base_model = MODELS[self.arch]()
        url = MODEL_URLS[(self.weights, self.arch)]
        # state_dict = model_zoo.load_url(url)
        # base_model.load_state_dict(state_dict)
        model = base_model.features

        modules = OrderedDict()
        if self.internal_preprocessing:
            modules["preprocessing"] = get_preprocessor(self.weights)

        block = depth = 1
        for module in model.children():
            if isinstance(module, nn.Conv2d):
                name = f"conv_{block}_{depth}"
            elif isinstance(module, nn.BatchNorm2d):
                name = f"bn_{block}_{depth}"
            elif isinstance(module, nn.ReLU):
                if not self.allow_inplace:
                    module = nn.ReLU(inplace=False)
                name = f"relu_{block}_{depth}"
                # each ReLU layer increases the depth of the current block
                depth += 1
            else:  # isinstance(module, nn.MaxPool2d):
                name = f"pool_{block}"
                # each pooling layer marks the end of the current block
                block += 1
                depth = 1

            modules[name] = module

        return modules

    def extra_repr(self):
        extras = [f"arch={self.arch}, " f"weights={self.weights}"]
        if not self.internal_preprocessing:
            extras.append(f"internal_preprocessing={self.internal_preprocessing}")
        if self.allow_inplace:
            extras.append(f"allow_inplace={self.allow_inplace}")
        return ", ".join(extras)


def _vgg_encoder(
    arch: str,
    weights: str = "torch",
    preprocessing: bool = True,
    allow_inplace: bool = False,
) -> VGGEncoder:
    return VGGEncoder(arch, weights, preprocessing, allow_inplace)


def vgg11_encoder(**kwargs) -> VGGEncoder:
    return _vgg_encoder("vgg11", **kwargs)


def vgg11_bn_encoder(**kwargs) -> VGGEncoder:
    return _vgg_encoder("vgg11_bn", **kwargs)


def vgg13_encoder(**kwargs) -> VGGEncoder:
    return _vgg_encoder("vgg13", **kwargs)


def vgg13_bn_encoder(**kwargs) -> VGGEncoder:
    return _vgg_encoder("vgg13_bn", **kwargs)


def vgg16_encoder(**kwargs) -> VGGEncoder:
    return _vgg_encoder("vgg16", **kwargs)


def vgg16_bn_encoder(**kwargs) -> VGGEncoder:
    return _vgg_encoder("vgg16_bn", **kwargs)


def vgg19_encoder(**kwargs) -> VGGEncoder:
    return _vgg_encoder("vgg19", **kwargs)


def vgg19_bn_encoder(**kwargs) -> VGGEncoder:
    return _vgg_encoder("vgg19_bn", **kwargs)
