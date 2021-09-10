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


@pytest.fixture
def mock_execution_with(mock_image_optimization, mock_write_image, set_argv):
    mock_image_optimization()
    mock_write_image()
    return set_argv


@contextlib.contextmanager
def exits(*, should_succeed=True, expected_code=None, check_err=None, check_out=None):
    def parse_checker(checker):
        if checker is None or callable(checker):
            return checker

        if isinstance(checker, str):
            checker = (checker,)

        def check_fn(text):
            for phrase in checker:
                assert phrase in text

        return check_fn

    check_err = parse_checker(check_err)
    check_out = parse_checker(check_out)

    with pytest.raises(SystemExit) as info:
        with contextlib.redirect_stderr(io.StringIO()) as raw_err:
            with contextlib.redirect_stdout(io.StringIO()) as raw_out:
                yield

    returned_code = info.value.code or 0
    succeeded = returned_code == 0
    err = raw_err.getvalue().strip()
    out = raw_out.getvalue().strip()

    if expected_code is not None:
        if returned_code == expected_code:
            return

        raise AssertionError(
            f"Returned and expected return code mismatch: "
            f"{returned_code} != {expected_code}."
        )

    if should_succeed:
        if succeeded:
            if check_out:
                check_out(out)

            return

        raise AssertionError(
            f"Program should have succeeded, but returned code {returned_code} "
            f"and printed the following to STDERR: '{err}'."
        )
    else:
        if not succeeded:
            if check_err:
                check_err(err)

            return

        raise AssertionError("Program shouldn't have succeeded, but did.")


@pytest.mark.parametrize("option", ["-h", "--help"])
def test_help_smoke(option, mock_execution_with):
    mock_execution_with(option)

    def check_out(out):
        assert out

    with exits(check_out=check_out):
        main()


@pytest.mark.parametrize("option", ["-V", "--version"])
def test_version(option, mock_execution_with):
    mock_execution_with(option)

    def check_out(out):
        assert out == pystiche.__version__

    with exits(check_out=check_out):
        main()


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
    def test_smoke(self, mock_execution_with, option):
        # If the output file is not specified,
        # we would get output to STDOUT regardless of -v / --verbose
        mock_execution_with(option, "--output-image=foo.jpg")

        def check_out(out):
            assert out

        with exits(check_out=check_out):
            main()

    def test_device(self, mock_execution_with):
        device = "cpu"
        mock_execution_with("--verbose", f"--device={device}")

        with exits(check_out=device):
            main()

    def test_mle(self, mock_execution_with):
        mle = "vgg19"
        mock_execution_with("--verbose", f"--multi-layer-encoder={mle}")

        with exits(check_out=("VGGMultiLayerEncoder", mle)):
            main()

    def test_perceptual_loss(self, mock_execution_with):
        content_loss = "FeatureReconstruction"
        style_loss = "Gram"

        mock_execution_with(
            "--verbose", f"--content-loss={content_loss}", f"--style-loss={style_loss}",
        )

        with exits(check_out=("PerceptualLoss", content_loss, style_loss)):
            main()


class TestDevice:
    def test_smoke(self, mock_execution_with):
        mock_execution_with("--device=cpu")

        with exits():
            main()

    def test_unknown(self, mock_execution_with):
        device = "unknown_device_type"
        mock_execution_with(f"--device={device}")

        with exits(should_succeed=False, check_err=device):
            main()

    def test_not_available(self, mock_execution_with):
        # hopefully no one ever has this available when running this test
        device = "mkldnn"
        mock_execution_with(f"--device={device}")

        with exits(should_succeed=False, check_err=device):
            main()


class TestMLE:
    def test_smoke(self, mock_execution_with):
        mock_execution_with("--mle=vgg19")

        with exits():
            main()

    @pytest.mark.parametrize(
        "mle",
        (
            pytest.param("vgg18", id="near_match"),
            pytest.param("unknown_multi_layer_encoder", id="unkonwn"),
        ),
    )
    def test_unknown(self, mock_execution_with, mle):
        mock_execution_with(f"--mle={mle}")

        with exits(should_succeed=False, check_err=mle):
            main()
