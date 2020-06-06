import pytest

import torch


class MarkConfig:
    def __init__(
        self,
        keyword,
        run_by_default,
        addoption=True,
        option=None,
        help=None,
        condition_for_skip=None,
        reason=None,
    ):
        self.addoption = addoption

        if option is None:
            option = (
                f"--{'skip' if run_by_default else 'run'}-{keyword.replace('_', '-')}"
            )
        self.option = option

        if help is None:
            help = (
                f"{'Skip' if run_by_default else 'Run'} tests "
                f"decorated with @{keyword}."
            )
        self.help = help

        if condition_for_skip is None:

            def condition_for_skip(config, item):
                has_keyword = keyword in item.keywords
                if run_by_default:
                    return has_keyword and config.getoption(option)
                else:
                    return has_keyword and not config.getoption(option)

        self.condition_for_skip = condition_for_skip

        if reason is None:
            reason = (
                f"Test is {keyword} and {option} was "
                f"{'' if run_by_default else 'not '}given."
            )
        self.marker = pytest.mark.skip(reason=reason)


MARK_CONFIGS = (
    MarkConfig(
        keyword="large_download",
        run_by_default=True,
        reason=(
            "Test possibly includes a large download and --skip-large-download was "
            "given."
        ),
    ),
    MarkConfig(
        keyword="slow",
        run_by_default=True,
        help=(
            "Skip tests decorated with @slow or "
            "@slow_if_cuda_not_available if CUDA is not available."
        ),
    ),
    MarkConfig(
        keyword="slow_if_cuda_not_available",
        run_by_default=True,
        addoption=False,
        condition_for_skip=(
            lambda config, item: (
                "slow_if_cuda_not_available" in item.keywords
                and config.getoption("--skip-slow")
                and not torch.cuda.is_available()
            )
        ),
        reason="Test is slow since CUDA is not available and --skip-slow was given.",
    ),
    MarkConfig(keyword="flaky", run_by_default=False),
)


def pytest_addoption(parser):
    for mark_config in MARK_CONFIGS:
        if mark_config.addoption:
            parser.addoption(
                mark_config.option,
                action="store_true",
                default=False,
                help=mark_config.help,
            )


def pytest_collection_modifyitems(config, items):
    for item in items:
        for mark_config in MARK_CONFIGS:
            if mark_config.condition_for_skip(config, item):
                item.add_marker(mark_config.marker)
