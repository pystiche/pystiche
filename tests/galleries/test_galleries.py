import os
import re
import unittest.mock
import warnings
from os import path

import pytest

from torchvision.models.vgg import _vgg

from tests import mocks, utils


def extract_sphinx_gallery_config():
    filters = []
    with unittest.mock.patch.object(warnings, "filters", filters):
        with utils.temp_add_to_sys_path(path.join("docs", "source")):
            sphinx_config = __import__("conf")

    return sphinx_config.sphinx_gallery_conf, filters


def collect_sphinx_gallery_scripts():
    sphinx_gallery_config, filters = extract_sphinx_gallery_config()
    file_pattern = re.compile(
        sphinx_gallery_config["filename_pattern"][1:] + r"[^.]*.py$"
    )

    dirs = set()
    scripts = []
    for root, _, files in os.walk(sphinx_gallery_config["examples_dirs"]):
        for file in files:
            match = file_pattern.match(file)
            if match is None:
                continue

            dirs.add(root)
            scripts.append(path.splitext(file)[0])

    return dirs, scripts, filters


DIRS, SCRIPTS, FILTERS = collect_sphinx_gallery_scripts()


@pytest.fixture(scope="module", autouse=True)
def add_gallery_dirs_to_sys_path():
    with utils.temp_add_to_sys_path(*DIRS, root=None):
        yield


@pytest.fixture(scope="package")
def sphinx_gallery_config(sphinx_config):
    return sphinx_config["sphinx_gallery_conf"]


@pytest.fixture(scope="module", autouse=True)
def patch_models_load_state_dict_from_url(package_mocker):
    mocks.patch_models_load_state_dict_from_url(mocker=package_mocker)

    # Since the beginner example "NST without pystiche" does not use a builtin
    # multi-layer encoder we are patching the model loader inplace.
    vgg_loader = _vgg

    def patched_vgg_loader(arch, cfg, batch_norm, pretrained, progress, **kwargs):
        return vgg_loader(arch, cfg, batch_norm, False, progress, **kwargs)

    package_mocker.patch(
        mocks.make_mock_target("models", "vgg", "_vgg", pkg="torchvision"),
        new=patched_vgg_loader,
    )


@pytest.fixture(scope="module", autouse=True)
def patch_matplotlib_figures(package_mocker):
    package_mocker.patch(mocks.make_mock_target("image", "show_image"))
    package_mocker.patch(
        mocks.make_mock_target("pyplot", "new_figure_manager", pkg="matplotlib")
    )


@pytest.fixture(autouse=True)
def patch_optimization(mocker):
    def patch_image_optimization():
        def image_optimization_side_effect(input_image, criterion, *args, **kwargs):
            criterion(input_image)
            return input_image

        for name in (
            "image_optimization",
            "pyramid_image_optimization",
        ):
            mocker.patch(
                mocks.make_mock_target("optim", name),
                side_effect=image_optimization_side_effect,
            )

    def patch_model_optimization():
        def model_optimization_side_effect(
            image_loader, transformer, criterion, *args, **kwargs
        ):
            input_image = next(image_loader)
            criterion(input_image)
            return transformer

        for name in ("model_optimization", "multi_epoch_model_optimization"):
            mocker.patch(
                mocks.make_mock_target("optim", name),
                side_effect=model_optimization_side_effect,
            )

    def patch_optimizer_step():
        orig_loss = None

        def lbfgs_step_side_effect(closure):
            nonlocal orig_loss
            if orig_loss is not None:
                return orig_loss

            orig_loss = closure()
            return orig_loss

        mocker.patch(
            mocks.make_mock_target("optim", "LBFGS", "step", pkg="torch"),
            side_effect=lbfgs_step_side_effect,
        )

    patch_image_optimization()
    patch_model_optimization()

    # Since the beginner example "NST without pystiche" does not use a builtin
    # optimization function we are patching the optimizer. The actual computation
    # happens inside a closure. Thus, the loop will run, albeit with an almost empty
    # body.
    patch_optimizer_step()


def would_be_filtered(warning):
    text = str(warning.message)
    category = warning.category
    module = warning.filename
    lineno = warning.lineno

    # taken from warnings.warn_explicit
    for item in FILTERS:
        action, msg, cat, mod, ln = item
        print()
        if (
            (msg is None or msg.match(text))
            and issubclass(category, cat)
            and (mod is None or mod.match(module))
            and (ln == 0 or lineno == ln)
        ):
            return True
    return False


def assert_no_warnings(recorder, script):
    warnings = [
        f"{warning.lineno}: {warning.category.__name__}: {warning.message}"
        for warning in recorder
        if not would_be_filtered(warning)
    ]
    msg = f"The execution of '{script}' emitted the following warnings:\n"
    msg += "\n".join(
        [
            f"{warning.lineno}: {warning.category.__name__}: {warning.message}"
            for warning in recorder
        ]
    )
    assert not warnings, msg


@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS)
def test_gallery_scripts_smoke(recwarn, script):
    __import__(script)
    assert_no_warnings(recwarn, script)
