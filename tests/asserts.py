import contextlib
import logging

from pystiche import optim

__all__ = [
    "assert_modules_identical",
    "assert_named_modules_identical",
    "assert_logs",
    "assert_property_in_repr",
]


def assert_modules_identical(actual, desired, equality_sufficient=False):
    actual = tuple(actual)
    desired = tuple(desired)

    for actual_module, desired_module in zip(actual, desired):
        if equality_sufficient:
            assert actual_module == desired_module
        else:
            assert actual_module is desired_module


def assert_named_modules_identical(actual, desired, equality_sufficient=False):
    actual_names, actual_modules = zip(*actual)
    desired_names, desired_modules = zip(*desired)

    assert actual_names == desired_names
    assert_modules_identical(
        actual_modules, desired_modules, equality_sufficient=equality_sufficient
    )


@contextlib.contextmanager
def assert_logs(caplog, level=logging.INFO, logger=None):
    if isinstance(logger, optim.OptimLogger):
        logger = logger.logger
    if isinstance(logger, logging.Logger):
        logger = logger.name

    with caplog.at_level(level, logger=logger):
        yield

    assert any(record.levelno == level for record in caplog.records)


def assert_property_in_repr(repr, name, value):
    assert f"{name}={value}" in repr
