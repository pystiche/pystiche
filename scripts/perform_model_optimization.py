import contextlib
import os
import sys
from collections import OrderedDict
from os import path
from unittest import mock

import torch

from pystiche.data import ImageFolderDataset
from pystiche.image import transforms


def main(root="."):
    mle, transformer, train = import_example()

    reload_mle(mle)
    dataset = make_dataset(root)
    train(transformer, dataset)

    state_dict = OrderedDict(
        [
            (name, parameter.detach().cpu())
            for name, parameter in transformer.state_dict().items()
        ]
    )
    torch.save(state_dict, "example_transformer.pth")


def import_example():
    with add_example_to_sys_path(), disable_io():
        import example_model_optimization as example

    return example.multi_layer_encoder, example.transformer, example.train


@contextlib.contextmanager
def add_example_to_sys_path():
    old = sys.path
    new = old.copy()
    sys.path = new

    here = path.dirname(__file__)
    project_root = path.abspath(path.join(here, ".."))
    example_dir = path.join(project_root, "examples", "advanced")
    new.insert(0, example_dir)

    try:
        yield
    finally:
        sys.path = old


@contextlib.contextmanager
def disable_io():
    with contextlib.ExitStack() as stack, open(os.devnull, "w") as devnull:
        stack.enter_context(contextlib.redirect_stdout(devnull))
        stack.enter_context(contextlib.redirect_stderr(devnull))

        for target in (
            "torch.nn.Module.load_state_dict",
            "torch.hub.load_state_dict_from_url",
            "pystiche.image.io._show_pil_image",
        ):
            stack.enter_context(mock.patch(target))

        yield


def reload_mle(mle):
    mle.load_state_dict_from_url(mle.framework, strict=False)


def make_dataset(root, image_size=256):
    transform = transforms.ComposedTransform(
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        OptionalGrayscaleToFakeGrayscale(),
    )
    return ImageFolderDataset(root, transform=transform)


class OptionalGrayscaleToFakeGrayscale(transforms.Transform):
    def forward(self, input):
        num_channels = input.size()[0]
        if num_channels == 1:
            return input.repeat(3, 1, 1)

        return input


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(
            "Please supply the root of the dataset as positional argument"
        )
    main(root=sys.argv[1])
