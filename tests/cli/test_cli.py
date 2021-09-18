import contextlib
import io
import pathlib
import sys
from collections import UserDict

import pytest

import pystiche
from pystiche import _cli as cli
from pystiche import demo

from tests.mocks import make_mock_target


@pytest.fixture(scope="module", autouse=True)
def cache_mle_loading(module_mocker):
    class CachedMLEDict(UserDict):
        def __init__(self, *args, **kwargs):
            self._mles = dict()
            super().__init__(*args, **kwargs)

        def __getitem__(self, name):
            if name not in self._mles:
                mle_fn = self.data[name]
                self._mles[name] = mle_fn()

            return lambda: self._mles[name]

    module_mocker.patch.object(cli, "MLES", new=CachedMLEDict(cli.MLES))


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

    with exits(check_out=""):
        cli.main()


@pytest.mark.parametrize("option", ["-V", "--version"])
def test_version(option, mock_execution_with):
    mock_execution_with(option)

    with exits(check_out=pystiche.__version__):
        cli.main()


@pytest.mark.slow
def test_smoke(mock_image_optimization, mock_write_image, set_argv):
    optim_mock = mock_image_optimization()
    io_mock = mock_write_image()
    set_argv()

    with exits():
        cli.main()

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
            cli.main()

    def test_device(self, mock_execution_with):
        device = "cpu"
        mock_execution_with("--verbose", f"--device={device}")

        with exits(check_out=device):
            cli.main()

    def test_mle(self, mock_execution_with):
        mle = "vgg19"
        mock_execution_with("--verbose", f"--multi-layer-encoder={mle}")

        with exits(check_out=("VGGMultiLayerEncoder", mle)):
            cli.main()

    def test_perceptual_loss(self, mock_execution_with):
        content_loss = "FeatureReconstruction"
        style_loss = "Gram"

        mock_execution_with(
            "--verbose", f"--content-loss={content_loss}", f"--style-loss={style_loss}",
        )

        with exits(check_out=("PerceptualLoss", content_loss, style_loss)):
            cli.main()


class TestDevice:
    def test_smoke(self, mock_execution_with):
        mock_execution_with("--device=cpu")

        with exits():
            cli.main()

    def test_unknown(self, mock_execution_with):
        device = "unknown_device_type"
        mock_execution_with(f"--device={device}")

        with exits(should_succeed=False, check_err=device):
            cli.main()

    def test_not_available(self, mock_execution_with):
        # hopefully no one ever has this available when running this test
        device = "mkldnn"
        mock_execution_with(f"--device={device}")

        with exits(should_succeed=False, check_err=device):
            cli.main()


class TestMLE:
    def test_smoke(self, mock_execution_with):
        mock_execution_with("--mle=vgg19")

        with exits():
            cli.main()

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
            cli.main()


class TestImage:
    options = pytest.mark.parametrize("option", ("content", "style", "starting_point"))

    @staticmethod
    def _mock_execution(mocker, *, option, value):
        if option == "content":
            args, kwargs = (), dict(content_image=value)
        elif option == "style":
            args, kwargs = (), dict(style_image=value)
        else:  # option == "starting_point"
            args, kwargs = (f"--starting-point={value}",), dict()
        mocker(*args, **kwargs)

    @pytest.mark.parametrize(
        "name",
        (
            pytest.param("bird", id="near_match"),
            pytest.param("unknown_demo_image", id="unkonwn"),
        ),
    )
    @options
    def test_unknown_demo_images(self, mock_execution_with, option, name):
        self._mock_execution(mock_execution_with, option=option, value=name)

        with exits(should_succeed=False, check_err=name):
            cli.main()

    @options
    def test_file_smoke(self, mock_execution_with, option):
        image = demo.images()["bird1"]
        image.download()
        file = pathlib.Path(pystiche.home()) / image.file
        self._mock_execution(mock_execution_with, option=option, value=str(file))

        with exits():
            cli.main()

    @options
    def test_non_existing_file(self, mock_execution_with, option):
        file = "/unknown/file.jpg"
        self._mock_execution(mock_execution_with, option=option, value=file)

        with exits(should_succeed=False, check_err=file):
            cli.main()


class TestLoss:
    def test_smoke(self, mock_execution_with):
        mock_execution_with("--content-loss=FeatureReconstruction")

        with exits():
            cli.main()

    @pytest.mark.parametrize(
        "loss",
        (
            pytest.param("FeaturReconstruction", id="near_match-FeatureReconstruction"),
            pytest.param("Gramm", id="near_match-Gram"),
            pytest.param("NRF", id="near_match-MRF"),
            pytest.param("unknown_loss", id="unkonwn"),
        ),
    )
    def test_unknown(self, mock_execution_with, loss):
        mock_execution_with(f"--content-loss={loss}")

        with exits(should_succeed=False, check_err=loss):
            cli.main()


class TestLayer:
    def test_smoke(self, mock_execution_with):
        mock_execution_with("--content-layer=relu4_2")

        with exits():
            cli.main()

    @pytest.mark.parametrize(
        "layer",
        [
            pytest.param("relu_4_2", id="near_match-single"),
            pytest.param("unknown_layer", id="unknown-single"),
            pytest.param("relu4_1,relu_4_2", id="near_match-multi"),
            pytest.param("relu4_1, unknown_layer", id="unknown-multi"),
        ],
    )
    def test_unknown(self, mock_execution_with, layer):
        mock_execution_with("--mle=vgg19, " f"--content-layer={layer}")

        with exits(should_succeed=False, check_err=layer):
            cli.main()
