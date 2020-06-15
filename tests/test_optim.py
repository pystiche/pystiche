import contextlib
import logging
import sys
from datetime import datetime, timedelta
from os import path

import pytest

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

import pystiche
from pystiche import optim

from .utils import PysticheTestCase, get_tmp_dir


class TestLog(PysticheTestCase):
    def test_default_logger_log_file(self):
        with get_tmp_dir() as tmp_dir:
            log_file = path.join(tmp_dir, "log_file.txt")
            logger = optim.default_logger(log_file=log_file)

            msg = "test message"
            logger.info(msg)

            with open(log_file, "r") as fh:
                lines = fh.readlines()

            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].strip().endswith(msg))

            # Windows compatibility
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.stream.close()

    def test_default_image_optim_log_fn_loss_dict_smoke(self):
        class MockOptimLogger:
            def __init__(self):
                self.msg = None

            @contextlib.contextmanager
            def environment(self, header):
                yield

            def message(self, msg):
                self.msg = msg

        loss_dict = pystiche.LossDict(
            (("a", torch.tensor(0.0)), ("b.c", torch.tensor(1.0)))
        )

        log_freq = 1
        max_depth = 1
        optim_logger = MockOptimLogger()
        log_fn = optim.default_image_optim_log_fn(
            optim_logger, log_freq=log_freq, max_depth=max_depth
        )

        step = log_freq
        log_fn(step, loss_dict)

        actual = optim_logger.msg
        desired = str(loss_dict.aggregate(max_depth))
        self.assertEqual(actual, desired)

    def test_default_image_optim_log_fn_other(self):
        optim_logger = optim.OptimLogger()
        log_freq = 1
        log_fn = optim.default_image_optim_log_fn(optim_logger, log_freq=log_freq)

        with self.assertRaises(TypeError):
            step = log_freq
            loss = None
            log_fn(step, loss)


class TestMeter(PysticheTestCase):
    class TestFloatMeter(optim.FloatMeter):
        def __init__(self, window_size=10):
            super().__init__("float_meter", window_size=window_size)

        def __str__(self):
            pass

    def test_FloatMeter_count(self):
        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter()
        meter.update(vals.tolist())

        actual = meter.count
        desired = len(vals)
        self.assertEqual(actual, desired)

    def test_FloatMeter_last_val(self):
        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter()
        meter.update(vals.tolist())

        actual = meter.last_val
        desired = vals[-1].item()
        self.assertFloatAlmostEqual(actual, desired)

    def test_FloatMeter_global_sum(self):
        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter()
        meter.update(vals.tolist())

        actual = meter.global_sum
        desired = torch.sum(vals).item()
        self.assertFloatAlmostEqual(actual, desired, decimal=6)

    def test_FloatMeter_global_min(self):
        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter()
        meter.update(vals.tolist())

        actual = meter.global_min
        desired = torch.min(vals).item()
        self.assertFloatAlmostEqual(actual, desired)

    def test_FloatMeter_global_max(self):
        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter()
        meter.update(vals.tolist())

        actual = meter.global_max
        desired = torch.max(vals).item()
        self.assertFloatAlmostEqual(actual, desired)

    def test_FloatMeter_global_avg(self):
        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter()
        meter.update(vals.tolist())

        actual = meter.global_avg
        desired = torch.mean(vals).item()
        self.assertFloatAlmostEqual(actual, desired, decimal=6)

    def test_FloatMeter_global_avg_empty(self):
        meter = self.TestFloatMeter()
        with self.assertRaises(RuntimeError):
            meter.global_avg

    def test_FloatMeter_local_avg(self):
        window_size = 5

        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter(window_size=window_size)
        meter.update(vals.tolist())

        actual = meter.local_avg
        desired = torch.mean(vals[-window_size:]).item()
        self.assertFloatAlmostEqual(actual, desired, decimal=6)

    def test_FloatMeter_local_avg_empty(self):
        meter = self.TestFloatMeter()
        with self.assertRaises(RuntimeError):
            meter.local_avg

    def test_AverageMeter_update_tensor(self):
        actual_meter = optim.AverageMeter("actual_meter")
        desired_meter = optim.AverageMeter("desired_meter")

        torch.manual_seed(0)
        vals = torch.rand(10)

        actual_meter.update(vals)
        desired_meter.update(vals.tolist())

        for attr in (
            "count",
            "last_val",
            "global_sum",
            "global_min",
            "global_max",
            "global_avg",
            "local_avg",
        ):
            actual = getattr(actual_meter, attr)
            desired = getattr(desired_meter, attr)
            self.assertAlmostEqual(actual, desired)

    def test_AverageMeter_str_smoke(self):
        for show_local_avg in (True, False):
            meter = optim.AverageMeter(
                "test_average_meter", show_local_avg=show_local_avg
            )
            self.assertIsInstance(str(meter), str)
            meter.update(0.0)
            self.assertIsInstance(str(meter), str)

    def test_LossMeter_update_LossDict(self):
        actual_meter = optim.LossMeter("actual_meter")
        desired_meter = optim.LossMeter("desired_meter")

        losses = torch.arange(3, dtype=torch.float)
        loss_dict = pystiche.LossDict(
            [(str(idx), loss) for idx, loss in enumerate(losses)]
        )

        actual_meter.update(loss_dict)
        desired_meter.update(torch.sum(losses).item())

        for attr in (
            "count",
            "last_val",
            "global_sum",
            "global_min",
            "global_max",
            "global_avg",
            "local_avg",
        ):
            actual = getattr(actual_meter, attr)
            desired = getattr(desired_meter, attr)
            self.assertAlmostEqual(actual, desired)

    def test_ETAMeter(self):
        first_time = 1.0
        second_time = 3.0
        meter = optim.ETAMeter(1)

        meter.update(first_time)

        actual = meter.count
        desired = 0
        self.assertEqual(actual, desired)

        actual = meter.last_time
        desired = first_time
        self.assertEqual(actual, desired)

        meter.update(second_time)

        actual = meter.count
        desired = 1
        self.assertEqual(actual, desired)

        actual = meter.last_time
        desired = second_time
        self.assertEqual(actual, desired)

        actual = meter.last_val
        desired = second_time - first_time
        self.assertEqual(actual, desired)

    def test_ETAMeter_calculate_eta(self):
        meter = optim.ETAMeter(1)
        time_diff = 10.0

        actual = meter.calculate_eta(time_diff)
        desired = datetime.now() + timedelta(seconds=time_diff)
        self.assertAlmostEqual(actual, desired, delta=timedelta(seconds=1.0))

        meter.update(1.0)
        meter.update(2.0)

        actual = meter.calculate_eta(time_diff)
        desired = datetime.now()
        self.assertAlmostEqual(actual, desired, delta=timedelta(seconds=1.0))

    def test_ETAMeter_str_smoke(self):
        for show_local_eta in (True, False):
            meter = optim.ETAMeter(1, show_local_eta=show_local_eta)
            self.assertIsInstance(str(meter), str)
            meter.update(0.0)
            self.assertIsInstance(str(meter), str)
            meter.update(0.0)
            self.assertIsInstance(str(meter), str)

    def test_ProgressMeter_update(self):
        meters = [optim.AverageMeter(str(idx)) for idx in range(3)]
        progress_meter = optim.ProgressMeter(1, *meters)

        vals = [float(val) for val in range(len(meters))]
        progress_meter.update(**{meter.name: val for meter, val in zip(meters, vals)})

        actual = progress_meter.count
        desired = 1
        self.assertEqual(actual, desired)

        for meter, val in zip(meters, vals):
            actual = meter.last_val
            desired = val
            self.assertFloatAlmostEqual(actual, desired)

    def test_ProgressMeter_str_smoke(self):
        for name in (None, "progress_meter"):
            meter = optim.ETAMeter(1, name=name)
            self.assertIsInstance(str(meter), str)
            meter.update(0.0)
            self.assertIsInstance(str(meter), str)


skip_if_py38 = pytest.mark.skipif(
    sys.version_info >= (3, 8),
    reason=(
        "Test errors on Python 3.8 only. This is most likely caused by the test "
        "itself rather than the code it should test."
    ),
)


class TestOptim(PysticheTestCase):
    def test_default_image_optimizer(self):
        torch.manual_seed(0)
        image = torch.rand(1, 3, 128, 128)
        optimizer = optim.default_image_optimizer(image)

        self.assertIsInstance(optimizer, Optimizer)

        actual = optimizer.param_groups[0]["params"][0]
        desired = image
        self.assertTensorAlmostEqual(actual, desired)

    @skip_if_py38
    def test_default_image_optim_loop(self):
        asset = self.load_asset(path.join("optim", "default_image_optim_loop"))

        actual = optim.default_image_optim_loop(
            asset.input.image,
            asset.input.criterion,
            get_optimizer=asset.params.get_optimizer,
            num_steps=asset.params.num_steps,
            quiet=True,
        )
        desired = asset.output.image
        self.assertTensorAlmostEqual(actual, desired, rtol=1e-4)

    @skip_if_py38
    def test_default_image_optim_loop_processing(self):
        asset = self.load_asset(
            path.join("optim", "default_image_optim_loop_processing")
        )

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
        self.assertTensorAlmostEqual(actual, desired, rtol=1e-4)

    @skip_if_py38
    def test_default_image_optim_loop_logging_smoke(self):
        asset = self.load_asset(path.join("optim", "default_image_optim_loop"))

        num_steps = 1
        optim_logger = optim.OptimLogger()
        log_fn = optim.default_image_optim_log_fn(optim_logger, log_freq=1)
        with self.assertLogs(optim_logger.logger, "INFO"):
            optim.default_image_optim_loop(
                asset.input.image,
                asset.input.criterion,
                num_steps=num_steps,
                log_fn=log_fn,
            )

    @skip_if_py38
    def test_default_image_pyramid_optim_loop(self):
        asset = self.load_asset(path.join("optim", "default_image_pyramid_optim_loop"))

        actual = optim.default_image_pyramid_optim_loop(
            asset.input.image,
            asset.input.criterion,
            asset.input.pyramid,
            get_optimizer=asset.params.get_optimizer,
            quiet=True,
        )
        desired = asset.output.image
        self.assertTensorAlmostEqual(actual, desired, rtol=1e-4)

    @skip_if_py38
    def test_default_image_pyramid_optim_loop_processing(self):
        asset = self.load_asset(path.join("optim", "default_image_pyramid_optim_loop"))

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
        self.assertTensorAlmostEqual(actual, desired, rtol=1e-4)

    @skip_if_py38
    def test_default_image_pyramid_optim_loop_logging_smoke(self):
        asset = self.load_asset(path.join("optim", "default_image_pyramid_optim_loop"))

        optim_logger = optim.OptimLogger()
        log_freq = max(level.num_steps for level in asset.input.pyramid._levels) + 1
        log_fn = optim.default_image_optim_log_fn(optim_logger, log_freq=log_freq)

        with self.assertLogs(optim_logger.logger, "INFO"):
            optim.default_image_pyramid_optim_loop(
                asset.input.image,
                asset.input.criterion,
                asset.input.pyramid,
                logger=optim_logger,
                log_fn=log_fn,
            )

    def test_default_transformer_optimizer(self):
        torch.manual_seed(0)
        transformer = nn.Conv2d(3, 3, 1)
        optimizer = optim.default_transformer_optimizer(transformer)

        self.assertIsInstance(optimizer, Optimizer)

        actuals = optimizer.param_groups[0]["params"]
        desireds = tuple(transformer.parameters())
        for actual, desired in zip(actuals, desireds):
            self.assertTensorAlmostEqual(actual, desired)

    @skip_if_py38
    def test_default_transformer_optim_loop(self):
        asset = self.load_asset(path.join("optim", "default_transformer_optim_loop"))

        transformer = asset.input.transformer
        optimizer = asset.params.get_optimizer(transformer)
        transformer = optim.default_transformer_optim_loop(
            asset.input.image_loader,
            transformer,
            asset.input.criterion,
            asset.input.criterion_update_fn,
            optimizer=optimizer,
            quiet=True,
        )

        actual = transformer.parameters()
        desired = asset.output.transformer.parameters()
        self.assertTensorSequenceAlmostEqual(actual, desired, rtol=1e-4)

    @skip_if_py38
    def test_default_transformer_optim_loop_logging_smoke(self):
        asset = self.load_asset(path.join("optim", "default_transformer_optim_loop"))

        image_loader = asset.input.image_loader
        optim_logger = optim.OptimLogger()
        log_fn = optim.default_transformer_optim_log_fn(
            optim_logger, len(image_loader), log_freq=1
        )

        with self.assertLogs(optim_logger.logger, "INFO"):
            optim.default_transformer_optim_loop(
                image_loader,
                asset.input.transformer,
                asset.input.criterion,
                asset.input.criterion_update_fn,
                logger=optim_logger,
                log_fn=log_fn,
            )

    @skip_if_py38
    def test_default_transformer_epoch_optim_loop(self):
        asset = self.load_asset(
            path.join("optim", "default_transformer_epoch_optim_loop")
        )

        transformer = asset.input.transformer
        optimizer = asset.params.get_optimizer(transformer)
        lr_scheduler = asset.params.get_lr_scheduler(optimizer)
        transformer = optim.default_transformer_epoch_optim_loop(
            asset.input.image_loader,
            transformer,
            asset.input.criterion,
            asset.input.criterion_update_fn,
            asset.input.epochs,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            quiet=True,
        )

        actual = transformer.parameters()
        desired = asset.output.transformer.parameters()
        self.assertTensorSequenceAlmostEqual(actual, desired, rtol=1e-4)

    @skip_if_py38
    def test_default_transformer_epoch_optim_loop_logging_smoke(self):
        asset = self.load_asset(
            path.join("optim", "default_transformer_epoch_optim_loop")
        )

        image_loader = asset.input.image_loader
        log_freq = len(image_loader) + 1
        optim_logger = optim.OptimLogger()
        log_fn = optim.default_transformer_optim_log_fn(
            optim_logger, len(image_loader), log_freq=log_freq
        )

        with self.assertLogs(optim_logger.logger, "INFO"):
            optim.default_transformer_epoch_optim_loop(
                asset.input.image_loader,
                asset.input.transformer,
                asset.input.criterion,
                asset.input.criterion_update_fn,
                asset.input.epochs,
                logger=optim_logger,
                log_fn=log_fn,
            )
