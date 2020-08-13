import contextlib
import logging
import sys
from os import path

import pytest

import torch

import pystiche
from pystiche import optim


def test_default_logger_log_file(tmpdir):
    log_file = path.join(tmpdir, "log_file.txt")
    logger = optim.default_logger(log_file=log_file)

    msg = "test message"
    logger.info(msg)

    with open(log_file, "r") as fh:
        lines = fh.readlines()

    assert len(lines) == 1

    line = lines[0]
    assert msg in line

    if sys.platform.startswith("win"):
        for handler in logger.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and handler.baseFilename == log_file
            ):
                handler.stream.close()


def test_default_image_optim_log_fn_loss_dict_smoke():
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
    assert actual == desired


def test_default_image_optim_log_fn_other():
    optim_logger = optim.OptimLogger()
    log_freq = 1
    log_fn = optim.default_image_optim_log_fn(optim_logger, log_freq=log_freq)

    with pytest.raises(TypeError):
        step = log_freq
        loss = None
        log_fn(step, loss)
