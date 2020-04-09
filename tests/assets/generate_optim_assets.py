from copy import deepcopy
from os import path

import torch
from torch import nn
from torch.utils.data import DataLoader

from pystiche.image import extract_aspect_ratio
from pystiche.image.transforms import CaffePostprocessing, CaffePreprocessing
from pystiche.ops import PixelComparisonOperator, TotalVariationOperator
from pystiche.optim import default_image_optimizer, default_transformer_optimizer
from pystiche.pyramid import ImagePyramid
from utils import store_asset


def _generate_default_image_optim_loop_asset(
    file,
    input_image,
    criterion,
    get_optimizer=None,
    num_steps=500,
    preprocessor=None,
    postprocessor=None,
):
    if get_optimizer is None:
        get_optimizer = default_image_optimizer

    output_image = input_image.clone()
    if preprocessor is not None:
        output_image = preprocessor(output_image)
    optimizer = get_optimizer(output_image)

    for step in range(num_steps):

        def closure():
            optimizer.zero_grad()

            loss = criterion(output_image)
            loss.backward()

            return loss

        optimizer.step(closure)

    output_image = output_image.detach()
    if postprocessor is not None:
        output_image = postprocessor(output_image)

    input = {"image": input_image, "criterion": criterion}
    params = {
        "get_optimizer": get_optimizer,
        "num_steps": num_steps,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
    }
    output = {"image": output_image}
    store_asset(input, params, output, file)


def generate_default_image_optim_loop_asset(root, file="default_image_optim_loop"):
    torch.manual_seed(0)
    input_image = torch.rand(1, 3, 32, 32)
    criterion = TotalVariationOperator()

    def get_optimizer(image):
        from torch.optim import Adam

        return Adam([image.requires_grad_(True)], lr=1e-1)

    num_steps = 5

    _generate_default_image_optim_loop_asset(
        path.join(root, file),
        input_image,
        criterion,
        get_optimizer=get_optimizer,
        num_steps=num_steps,
    )


def generate_default_image_optim_loop_processing_asset(
    root, file="default_image_optim_loop_processing"
):
    torch.manual_seed(0)
    input_image = torch.rand(1, 3, 32, 32)
    criterion = TotalVariationOperator()

    def get_optimizer(image):
        from torch.optim import Adam

        return Adam([image.requires_grad_(True)], lr=1e-1)

    num_steps = 5
    preprocessor = CaffePreprocessing()
    postprocessor = CaffePostprocessing()

    _generate_default_image_optim_loop_asset(
        path.join(root, file),
        input_image,
        criterion,
        get_optimizer=get_optimizer,
        num_steps=num_steps,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def _generate_default_image_pyramid_optim_loop_asset(
    file,
    input_image,
    criterion,
    pyramid,
    get_optimizer=None,
    preprocessor=None,
    postprocessor=None,
):
    if get_optimizer is None:
        get_optimizer = default_image_optimizer

    aspect_ratio = extract_aspect_ratio(input_image)
    output_image = input_image.clone()
    for level in pyramid:
        with torch.no_grad():
            output_image = level.resize_image(output_image, aspect_ratio=aspect_ratio)
        if preprocessor is not None:
            output_image = preprocessor(output_image)
        optimizer = get_optimizer(output_image)

        for _ in level:

            def closure():
                optimizer.zero_grad()

                loss = criterion(output_image)
                loss.backward()

                return loss

            optimizer.step(closure)

        output_image = output_image.detach()
        if postprocessor is not None:
            output_image = postprocessor(output_image)

    input = {"image": input_image, "criterion": criterion, "pyramid": pyramid}
    params = {
        "get_optimizer": get_optimizer,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
    }
    output = {"image": output_image}
    store_asset(input, params, output, file)


def generate_default_image_pyramid_optim_loop_asset(
    root, file="default_image_pyramid_optim_loop"
):
    torch.manual_seed(0)
    input_image = torch.rand(1, 3, 32, 32)
    criterion = TotalVariationOperator()
    pyramid = ImagePyramid((16, 32), 3)

    def get_optimizer(image):
        from torch.optim import Adam

        return Adam([image.requires_grad_(True)], lr=1e-1)

    _generate_default_image_pyramid_optim_loop_asset(
        path.join(root, file),
        input_image,
        criterion,
        pyramid,
        get_optimizer=get_optimizer,
    )


def generate_default_image_pyramid_optim_loop__processing_asset(
    root, file="default_image_pyramid_optim_loop_processing"
):
    torch.manual_seed(0)
    input_image = torch.rand(1, 3, 32, 32)
    criterion = TotalVariationOperator()
    pyramid = ImagePyramid((16, 32), 3)

    def get_optimizer(image):
        from torch.optim import Adam

        return Adam([image.requires_grad_(True)], lr=1e-1)

    preprocessor = CaffePreprocessing()
    postprocessor = CaffePostprocessing()

    _generate_default_image_pyramid_optim_loop_asset(
        path.join(root, file),
        input_image,
        criterion,
        pyramid,
        get_optimizer=get_optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def _generate_default_transformer_optim_loop_asset(
    file,
    image_loader,
    device,
    transformer,
    criterion,
    criterion_update_fn,
    get_optimizer=None,
):
    input_transformer = deepcopy(transformer)

    if get_optimizer is None:
        get_optimizer = default_transformer_optimizer
    optimizer = get_optimizer(transformer)

    for target_image in image_loader:
        target_image = target_image.to(device)
        criterion_update_fn(target_image, criterion)
        input_image = transformer(target_image)

        def closure():
            optimizer.zero_grad()

            loss = criterion(input_image)
            loss.backward()

            return loss

        optimizer.step(closure)

    input = {
        "image_loader": image_loader,
        "device": device,
        "transformer": input_transformer,
        "criterion": criterion,
        "criterion_update_fn": criterion_update_fn,
    }
    params = {
        "get_optimizer": get_optimizer,
    }
    output = {"transformer": transformer}
    store_asset(input, params, output, file)


def generate_default_transformer_optim_loop_asset(
    root, file="default_transformer_optim_loop"
):
    torch.manual_seed(0)
    image_loader = DataLoader([torch.rand(3, 32, 32) for _ in range(3)])
    device = torch.device("cpu")
    transformer = nn.Conv2d(3, 3, 1).train()

    class MSEOperator(PixelComparisonOperator):
        def target_image_to_repr(self, image):
            return image, None

        def input_image_to_repr(self, image, ctx):
            return image

        def calculate_score(self, input_repr, target_repr, ctx):
            from torch.nn.functional import mse_loss

            return mse_loss(input_repr, target_repr)

    criterion = MSEOperator().eval()

    def criterion_update_fn(target_image, criterion):
        criterion.set_target_image(target_image)

    def get_optimizer(transformer):
        from torch.optim import Adam

        return Adam(transformer.parameters(), lr=1e-1)

    _generate_default_transformer_optim_loop_asset(
        path.join(root, file),
        image_loader,
        device,
        transformer,
        criterion,
        criterion_update_fn,
        get_optimizer=get_optimizer,
    )


def _generate_default_transformer_epoch_optim_loop_asset(
    file,
    image_loader,
    transformer,
    criterion,
    criterion_update_fn,
    epochs,
    get_lr_scheduler,
    get_optimizer=None,
):
    input_transformer = deepcopy(transformer)

    if get_optimizer is None:
        get_optimizer = default_transformer_optimizer
    optimizer = get_optimizer(transformer)

    lr_scheduler = get_lr_scheduler(optimizer)

    for epoch in range(epochs):
        for target_image in image_loader:
            criterion_update_fn(target_image, criterion)
            input_image = transformer(target_image)

            def closure():
                optimizer.zero_grad()

                loss = criterion(input_image)
                loss.backward()

                return loss

            optimizer.step(closure)

        lr_scheduler.step()

    input = {
        "image_loader": image_loader,
        "transformer": input_transformer,
        "criterion": criterion,
        "criterion_update_fn": criterion_update_fn,
        "epochs": epochs,
    }
    params = {
        "get_optimizer": get_optimizer,
        "get_lr_scheduler": get_lr_scheduler,
    }
    output = {"transformer": transformer}
    store_asset(input, params, output, file)


def generate_default_transformer_epoch_optim_loop_asset(
    root, file="default_transformer_epoch_optim_loop"
):
    torch.manual_seed(0)
    image_loader = DataLoader([torch.rand(3, 32, 32) for _ in range(2)])
    transformer = nn.Conv2d(3, 3, 1).train()

    class MSEOperator(PixelComparisonOperator):
        def target_image_to_repr(self, image):
            return image, None

        def input_image_to_repr(self, image, ctx):
            return image

        def calculate_score(self, input_repr, target_repr, ctx):
            from torch.nn.functional import mse_loss

            return mse_loss(input_repr, target_repr)

    criterion = MSEOperator().eval()

    def criterion_update_fn(target_image, criterion):
        criterion.set_target_image(target_image)

    epochs = 3

    def get_optimizer(transformer):
        from torch.optim import Adam

        return Adam(transformer.parameters(), lr=1e-1)

    def get_lr_scheduler(optimizer):
        from torch.optim.lr_scheduler import ExponentialLR

        return ExponentialLR(optimizer, gamma=0.5)

    _generate_default_transformer_epoch_optim_loop_asset(
        path.join(root, file),
        image_loader,
        transformer,
        criterion,
        criterion_update_fn,
        epochs,
        get_optimizer=get_optimizer,
        get_lr_scheduler=get_lr_scheduler,
    )


def main(root):
    generate_default_image_optim_loop_asset(root)
    generate_default_image_optim_loop_processing_asset(root)

    generate_default_image_pyramid_optim_loop_asset(root)
    generate_default_image_pyramid_optim_loop__processing_asset(root)

    generate_default_transformer_optim_loop_asset(root)

    generate_default_transformer_epoch_optim_loop_asset(root)


if __name__ == "__main__":
    root = path.join(path.dirname(__file__), "optim")
    main(root)
