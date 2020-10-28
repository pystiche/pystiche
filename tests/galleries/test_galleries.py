import os
import re
from os import path

import pytest

from torchvision.models.vgg import _vgg

from tests import mocks, utils


def extract_sphinx_gallery_config():
    sphinx_config_file = path.abspath(
        path.join(path.dirname(__file__), "..", "..", "docs", "source", "conf.py")
    )
    sphinx_config, _ = utils.exec_file(sphinx_config_file)
    return sphinx_config["sphinx_gallery_conf"]


def collect_sphinx_gallery_scripts():
    sphinx_gallery_config = extract_sphinx_gallery_config()
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

    return dirs, scripts


DIRS, SCRIPTS = collect_sphinx_gallery_scripts()


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
    def optim_loop_side_effect(input, *args, **kwargs):
        return input

    for name in (
        "image_optimization",
        "pyramid_image_optimization",
        "model_optimization",
        "multi_epoch_model_optimization",
    ):
        mocker.patch(
            mocks.make_mock_target("optim", name), side_effect=optim_loop_side_effect
        )

    # Since the beginner example "NST without pystiche" does not use a builtin
    # optimization function we are patching the optimizer. The actual computation
    # happens inside a closure. Thus, the loop will run, albeit with an almost empty
    # body.
    mocker.patch(mocks.make_mock_target("optim", "LBFGS", pkg="torch"))


@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS)
def test_gallery_scripts_smoke(recwarn, script):
    __import__(script)

    msg = f"The execution of '{script}' emitted the following warnings:\n"
    msg += "\n".join(
        [
            f"{warning.lineno}: {warning.category.__name__}: {warning.message}"
            for warning in recwarn
        ]
    )
    assert not recwarn, msg
