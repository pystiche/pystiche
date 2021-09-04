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
    optim_mock = mock_image_optimization()
    io_mock = mock_write_image()
    set_argv()

    with exits():
        main()

    optim_mock.assert_called_once()
    io_mock.assert_called_once()


@pytest.mark.slow
class TestVerbose:
    @pytest.mark.parametrize("option", ["-v", "--verbose"])
    def test_smoke(self, mock_image_optimization, mock_write_image, set_argv, option):
        mock_image_optimization()
        mock_write_image()
        # If the output file is not specified,
        # we would get output to STDOUT regardless of -v / --verbose
        set_argv(option, "--output-image=foo.jpg")

        with contextlib.redirect_stdout(io.StringIO()) as output, exits():
            main()

        assert output.getvalue().strip()

    def test_device(self, mock_image_optimization, mock_write_image, set_argv):
        device = "cpu"

        mock_image_optimization()
        mock_write_image()
        set_argv("--verbose", f"--device={device}")

        with contextlib.redirect_stdout(io.StringIO()) as output, exits():
            main()

        assert "'cpu'" in output.getvalue().strip()

    def test_mle(self, mock_image_optimization, mock_write_image, set_argv):
        mle = "vgg19"

        mock_image_optimization()
        mock_write_image()
        set_argv("--verbose", f"--multi-layer-encoder={mle}")

        with contextlib.redirect_stdout(io.StringIO()) as output, exits():
            main()

        output = output.getvalue().strip()
        assert "VGGMultiLayerEncoder" in output
        assert f"arch={mle}" in output

    def test_perceptual_loss(self, mock_image_optimization, mock_write_image, set_argv):
        content_loss = "FeatureReconstruction"
        style_loss = "Gram"

        mock_image_optimization()
        mock_write_image()
        set_argv(
            "--verbose", f"--content-loss={content_loss}", f"--style-loss={style_loss}",
        )

        with contextlib.redirect_stdout(io.StringIO()) as output, exits():
            main()

        output = output.getvalue().strip()
        assert "PerceptualLoss" in output
        assert content_loss in output
        assert style_loss in output
