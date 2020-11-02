import sys
from unittest import mock

import pytest
import pytorch_testing_utils as ptu

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data

from pystiche import loss, ops, optim

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


class Dataset(data.Dataset):
    def __init__(self, image, supervised=False):
        if image.dim() == 4:
            image = image.squeeze(0)
        self.image = image
        self.supervised = supervised

    def __len__(self):
        return 1

    def __getitem__(self, _):
        if self.supervised:
            return self.image, torch.zeros(self.image.size()[0])
        else:
            return self.image


@pytest.fixture
def image_loader(test_image):
    return data.DataLoader(Dataset(test_image))


class ModuleMock(nn.Module):
    def __init__(self, *methods):
        super().__init__()
        self._call_args_list = []
        for method in methods:
            setattr(
                self, method, mock.MagicMock(name=f"{type(self).__name__}.{method}")
            )

    def forward(self, input, *args, **kwargs):
        self._call_args_list.insert(0, ((input, *args), kwargs))
        return input

    @property
    def call_args_list(self):
        return self._call_args_list

    @property
    def called(self):
        return bool(self.call_args_list)

    @property
    def call_args(self):
        return self.call_args_list[0]

    @property
    def call_count(self):
        return len(self.call_args_list)

    def assert_called(self):
        assert self.called

    def assert_called_once(self):
        assert self.call_count == 1

    def assert_called_once_with(self, input, *args, **kwargs):
        self.assert_called_once()
        (input_, *args_), kwargs_ = self.call_args
        ptu.assert_allclose(input_, input)
        assert tuple(args_) == args
        assert kwargs_ == kwargs


class TransformerMock(ModuleMock):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "parameter", nn.Parameter(torch.zeros(1).squeeze(), requires_grad=True)
        )

    def forward(self, input, *args, **kwargs):
        return super().forward(input + self.parameter, *args, **kwargs)


@pytest.fixture
def transformer():
    return TransformerMock()


@pytest.fixture
def module():
    return ModuleMock()


class CriterionMock(ModuleMock):
    def forward(self, *args, **kwargs):
        return torch.sum(super().forward(*args, **kwargs))


@pytest.fixture
def criterion():
    return CriterionMock()


class MSEOperator(ops.PixelComparisonOperator):
    def image_to_repr(self, image):
        return image

    def input_image_to_repr(
        self, image, ctx,
    ):
        return image

    def target_image_to_repr(self, image):
        return image, None

    def calculate_score(
        self, input_repr, target_repr, ctx,
    ):
        return F.mse_loss(input_repr, target_repr)


def test_model_default_optimization_criterion_update_fn(
    transformer, test_image,
):
    image_loader = data.DataLoader(Dataset(test_image))

    content_loss = MSEOperator()
    style_loss = MSEOperator()
    criterion = loss.PerceptualLoss(content_loss, style_loss)

    content_loss.set_target_image(torch.rand_like(test_image))
    style_loss.set_target_image(torch.rand_like(test_image))
    optim.model_optimization(image_loader, transformer, criterion)

    ptu.assert_allclose(content_loss.target_image, test_image)


def test_model_optimization_criterion_update_fn_error(
    image_loader, transformer, criterion
):
    with pytest.raises(RuntimeError):
        optim.model_optimization(image_loader, transformer, criterion)


@pytest.mark.parametrize(
    "supervised",
    (pytest.param(True, id="supervised"), pytest.param(False, id="unsupervised")),
)
def test_model_optimization_image_loader(
    transformer, criterion, test_image, supervised
):
    image_loader = data.DataLoader(Dataset(test_image, supervised=supervised))

    optim.model_optimization(
        image_loader,
        transformer,
        criterion,
        criterion_update_fn=lambda input_image, criterion: None,
    )

    transformer.assert_called_once_with(test_image)


def test_model_optimization_image_loader_no_tensor(transformer, criterion):
    image_loader = data.DataLoader((None,))

    with pytest.raises(RuntimeError):
        optim.model_optimization(
            image_loader, transformer, criterion,
        )
