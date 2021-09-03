import contextlib
import io
import sys

import pytest

import pystiche
from pystiche._cli import main


@pytest.fixture
def set_argv(mocker):
    def patch_argv_(*args):
        return mocker.patch.object(sys, "argv", ["pystiche", *args])

    return patch_argv_


@contextlib.contextmanager
def exits(*, code=None, error=False):
    with pytest.raises(SystemExit) as info:
        yield

    ret = info.value.code

    if code is not None:
        assert ret == code

    if error:
        assert ret >= 1
    else:
        assert ret is None or ret == 0


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
