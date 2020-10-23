import sys

import pytest
import pytorch_testing_utils as ptu

import torch
from torch import nn

from pystiche import optim

from tests import asserts

skip_if_py38 = pytest.mark.skipif(
    sys.version_info >= (3, 8),
    reason=(
        "Test errors on Python 3.8 only. This is most likely caused by the test "
        "itself rather than the code it should test."
    ),
)


def test_default_image_optimizer():
    torch.manual_seed(0)
    image = torch.rand(1, 3, 128, 128)
    optimizer = optim.default_image_optimizer(image)

    assert isinstance(optimizer, torch.optim.Optimizer)

    actual = optimizer.param_groups[0]["params"][0]
    desired = image
    ptu.assert_allclose(actual, desired)


def test_image_optimization_optimizer_preprocessor():
    input_image = torch.empty(1)
    criterion = nn.Module()
    optimizer = optim.default_image_optimizer(input_image)
    preprocessor = nn.Module()

    with pytest.raises(RuntimeError):
        optim.image_optimization(
            input_image, criterion, optimizer, preprocessor=preprocessor
        )


@skip_if_py38
def test_default_image_optim_loop(optim_asset_loader):
    asset = optim_asset_loader("default_image_optim_loop")

    actual = optim.default_image_optim_loop(
        asset.input.image,
        asset.input.criterion,
        get_optimizer=asset.params.get_optimizer,
        num_steps=asset.params.num_steps,
        quiet=True,
    )
    desired = asset.output.image
    ptu.assert_allclose(actual, desired, rtol=1e-4)


@skip_if_py38
def test_default_image_optim_loop_processing(optim_asset_loader):
    asset = optim_asset_loader("default_image_optim_loop_processing")

    actual = optim.default_image_optim_loop(
        asset.input.image,
        asset.input.criterion,
        get_optimizer=asset.params.get_optimizer,
        num_steps=asset.params.num_steps,
        preprocessor=asset.params.preprocessor,
        postprocessor=asset.params.postprocessor,
        quiet=True,
    )
    desired = asset.output.image
    ptu.assert_allclose(actual, desired, rtol=1e-4)


@skip_if_py38
def test_default_image_optim_loop_logging_smoke(caplog, optim_asset_loader):
    asset = optim_asset_loader("default_image_optim_loop")

    num_steps = 1
    optim_logger = optim.OptimLogger()
    log_fn = optim.default_image_optim_log_fn(optim_logger, log_freq=1)
    with asserts.assert_logs(caplog, logger=optim_logger):
        optim.default_image_optim_loop(
            asset.input.image,
            asset.input.criterion,
            num_steps=num_steps,
            log_fn=log_fn,
        )


@skip_if_py38
def test_default_image_pyramid_optim_loop(optim_asset_loader):
    asset = optim_asset_loader("default_image_pyramid_optim_loop")

    actual = optim.default_image_pyramid_optim_loop(
        asset.input.image,
        asset.input.criterion,
        asset.input.pyramid,
        get_optimizer=asset.params.get_optimizer,
        quiet=True,
    )
    desired = asset.output.image
    ptu.assert_allclose(actual, desired, rtol=1e-4)


@skip_if_py38
def test_default_image_pyramid_optim_loop_processing(optim_asset_loader):
    asset = optim_asset_loader("default_image_pyramid_optim_loop")

    actual = optim.default_image_pyramid_optim_loop(
        asset.input.image,
        asset.input.criterion,
        asset.input.pyramid,
        get_optimizer=asset.params.get_optimizer,
        preprocessor=asset.params.preprocessor,
        postprocessor=asset.params.postprocessor,
        quiet=True,
    )
    desired = asset.output.image
    ptu.assert_allclose(actual, desired, rtol=1e-4)


@skip_if_py38
def test_default_image_pyramid_optim_loop_logging_smoke(caplog, optim_asset_loader):
    asset = optim_asset_loader("default_image_pyramid_optim_loop")

    optim_logger = optim.OptimLogger()
    log_freq = max(level.num_steps for level in asset.input.pyramid._levels) + 1
    log_fn = optim.default_image_optim_log_fn(optim_logger, log_freq=log_freq)

    with asserts.assert_logs(caplog, logger=optim_logger):
        optim.default_image_pyramid_optim_loop(
            asset.input.image,
            asset.input.criterion,
            asset.input.pyramid,
            logger=optim_logger,
            log_fn=log_fn,
        )


def test_default_transformer_optimizer():
    torch.manual_seed(0)
    transformer = nn.Conv2d(3, 3, 1)
    optimizer = optim.default_transformer_optimizer(transformer)

    assert isinstance(optimizer, torch.optim.Optimizer)

    actuals = optimizer.param_groups[0]["params"]
    desireds = tuple(transformer.parameters())
    for actual, desired in zip(actuals, desireds):
        ptu.assert_allclose(actual, desired)


def make_torch_ge_1_6_compatible(image_loader, criterion):
    # See https://github.com/pmeier/pystiche/pull/348 for a discussion of this
    image_loader.generator = None

    for module in criterion.modules():
        module._non_persistent_buffers_set = set()


@skip_if_py38
def test_default_transformer_optim_loop(optim_asset_loader):
    asset = optim_asset_loader("default_transformer_optim_loop")

    image_loader = asset.input.image_loader
    criterion = asset.input.criterion
    make_torch_ge_1_6_compatible(image_loader, criterion)

    transformer = asset.input.transformer
    optimizer = asset.params.get_optimizer(transformer)
    transformer = optim.default_transformer_optim_loop(
        image_loader,
        transformer,
        criterion,
        asset.input.criterion_update_fn,
        optimizer=optimizer,
        quiet=True,
    )

    actual = tuple(transformer.parameters())
    desired = tuple(asset.output.transformer.parameters())
    ptu.assert_allclose(actual, desired, rtol=1e-4)


@skip_if_py38
def test_default_transformer_optim_loop_logging_smoke(caplog, optim_asset_loader):
    asset = optim_asset_loader("default_transformer_optim_loop")

    image_loader = asset.input.image_loader
    criterion = asset.input.criterion
    make_torch_ge_1_6_compatible(image_loader, criterion)

    optim_logger = optim.OptimLogger()
    log_fn = optim.default_transformer_optim_log_fn(
        optim_logger, len(image_loader), log_freq=1
    )

    with asserts.assert_logs(caplog, logger=optim_logger):
        optim.default_transformer_optim_loop(
            image_loader,
            asset.input.transformer,
            criterion,
            asset.input.criterion_update_fn,
            logger=optim_logger,
            log_fn=log_fn,
        )


@skip_if_py38
def test_default_transformer_epoch_optim_loop(optim_asset_loader):
    asset = optim_asset_loader("default_transformer_epoch_optim_loop")

    image_loader = asset.input.image_loader
    criterion = asset.input.criterion
    make_torch_ge_1_6_compatible(image_loader, criterion)

    transformer = asset.input.transformer
    optimizer = asset.params.get_optimizer(transformer)
    lr_scheduler = asset.params.get_lr_scheduler(optimizer)
    transformer = optim.default_transformer_epoch_optim_loop(
        image_loader,
        transformer,
        criterion,
        asset.input.criterion_update_fn,
        asset.input.epochs,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        quiet=True,
    )

    actual = tuple(transformer.parameters())
    desired = tuple(asset.output.transformer.parameters())
    ptu.assert_allclose(actual, desired, rtol=1e-4)


@skip_if_py38
def test_default_transformer_epoch_optim_loop_logging_smoke(caplog, optim_asset_loader):
    asset = optim_asset_loader("default_transformer_epoch_optim_loop")

    image_loader = asset.input.image_loader
    criterion = asset.input.criterion
    make_torch_ge_1_6_compatible(image_loader, criterion)

    log_freq = len(image_loader) + 1
    optim_logger = optim.OptimLogger()
    log_fn = optim.default_transformer_optim_log_fn(
        optim_logger, len(image_loader), log_freq=log_freq
    )

    with asserts.assert_logs(caplog, logger=optim_logger):
        optim.default_transformer_epoch_optim_loop(
            image_loader,
            asset.input.transformer,
            criterion,
            asset.input.criterion_update_fn,
            asset.input.epochs,
            logger=optim_logger,
            log_fn=log_fn,
        )
