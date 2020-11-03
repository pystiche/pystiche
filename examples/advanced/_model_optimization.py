import contextlib
import sys
from collections import OrderedDict
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
    @contextlib.contextmanager
    def disable():
        targets = (
            "torch.nn.Module.load_state_dict",
            "torch.hub.load_state_dict_from_url",
            "pystiche.image.io._show_pil_image",
        )
        with contextlib.ExitStack() as stack:
            for target in targets:
                stack.enter_context(mock.patch(target))
            yield

    with disable(), contextlib.redirect_stdout(None):
        import example_model_optimization as example

    return example.multi_layer_encoder, example.transformer, example.train


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
    root = sys.argv[1] if len(sys.argv) >= 2 else "."
    main(root=root)
