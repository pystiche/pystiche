import contextlib
import io
import sys

import pytest

import pystiche
from pystiche._cli import main

from tests.mocks import make_mock_target


@pytest.fixture
def mock_image_optimization(mocker):
    def mock():
        return mocker.patch(make_mock_target("_cli", "write_image"))

    return mock


@pytest.fixture
def mock_write_image(mocker):
    def mock():
        return mocker.patch(make_mock_target("_cli", "image_optimization"))

    return mock


@pytest.fixture
def set_argv(mocker):
    def set_argv_(*options, content_image="bird1", style_image="paint"):
        return mocker.patch.object(
            sys, "argv", ["pystiche", *options, content_image, style_image]
        )

    return set_argv_


@contextlib.contextmanager
def exits(*, should_succeed=True, expected_code=None):
    with pytest.raises(SystemExit) as info:
        with contextlib.redirect_stderr(io.StringIO()) as output:
            yield

    returned_code = info.value.code or 0
    succeeded = returned_code == 0

    if expected_code is not None:
        if returned_code == expected_code:
            return

        raise AssertionError(
            f"Returned and expected return code mismatch: "
            f"{returned_code} != {expected_code}."
        )

    if should_succeed:
        if succeeded:
            return

        raise AssertionError(
            f"Program should have succeeded, "
            f"but returned code {returned_code} and printed the following to STDERR: "
            f"'{output.getvalue().strip()}'."
        )
    else:
        if not succeeded:
            return

        raise AssertionError("Program shouldn't have succeeded, but did.")


@pytest.mark.parametrize("option", ["-h", "--help"])
def test_help_smoke(option, set_argv):
    set_argv(option)

    with contextlib.redirect_stdout(io.StringIO()) as output, exits():
        main()

    assert output.getvalue()


@pytest.mark.parametrize("option", ["-V", "--version"])
def test_version(option, set_argv):
    set_argv(option)

    with contextlib.redirect_stdout(io.StringIO()) as output, exits():
        main()

    assert output.getvalue().strip() == pystiche.__version__


@pytest.mark.slow
def test_smoke(mock_image_optimization, mock_write_image, set_argv):
    mock_image_optimization()
    mock_write_image()
    set_argv()

    with exits():
        main()
