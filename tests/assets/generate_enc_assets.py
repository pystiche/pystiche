from collections import OrderedDict
from os import path

import torch
from torch import nn
from torchvision import models

import pystiche
from utils import store_asset


def _generate_sequential_enc_asset(file, model, image, precision=2):
    model.eval()
    input_image = image.clone()
    enc_keys = {}
    for layer, module in model.named_children():
        image = module(image)
        enc_keys[layer] = pystiche.TensorKey(image, precision=precision)

    input = {"image": input_image}
    params = {"precision": precision}
    output = {"enc_keys": enc_keys}
    store_asset(input, params, output, file)


def generate_alexnet_asset(root, file="alexnet"):
    model = models.alexnet(pretrained=True).features
    modules = OrderedDict()
    block = 1
    for idx, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            layer = f"conv{block}"
        elif isinstance(module, nn.ReLU):
            layer = f"relu{block}"
            module = nn.ReLU(inplace=False)
            if block in (3, 4):
                block += 1
        elif isinstance(module, nn.MaxPool2d):
            layer = f"pool{block}"
            block += 1
        else:
            raise RuntimeError

        modules[layer] = module
    model = nn.Sequential(modules)

    torch.manual_seed(0)
    image = torch.rand(1, 3, 256, 256)

    _generate_sequential_enc_asset(path.join(root, file), model, image)


def generate_vgg_assets(root):
    archs = ("vgg11", "vgg13", "vgg16", "vgg19")
    archs = (*archs, *[f"{arch}_bn" for arch in archs])

    for arch in archs:
        model = models.__dict__[arch](pretrained=True).features
        modules = OrderedDict()

        block = depth = 1
        for idx, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                layer = f"conv{block}_{depth}"
            elif isinstance(module, nn.BatchNorm2d):
                layer = f"bn{block}_{depth}"
            elif isinstance(module, nn.ReLU):
                layer = f"relu{block}_{depth}"
                module = nn.ReLU(inplace=False)
                depth += 1
            elif isinstance(module, nn.MaxPool2d):
                layer = f"pool{block}"
                block += 1
                depth = 1
            else:
                raise RuntimeError

            modules[layer] = module
        model = nn.Sequential(modules)

        torch.manual_seed(0)
        image = torch.rand(1, 3, 256, 256)

        _generate_sequential_enc_asset(path.join(root, arch), model, image)


def main(root):
    generate_alexnet_asset(root)

    generate_vgg_assets(root)


if __name__ == "__main__":
    root = path.join(path.dirname(__file__), "enc")
    main(root)
