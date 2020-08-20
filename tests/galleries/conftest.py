import os
import re
from os import path

import pytest

from torchvision.models.vgg import _vgg

from tests import mocks, utils

SPHINX_CONFIG_FILE = path.abspath(
    path.join(path.dirname(__file__), "..", "..", "docs", "source", "conf.py")
)


@pytest.fixture(scope="package")
def sphinx_config():
    config, _ = utils.exec_file(SPHINX_CONFIG_FILE)
    return config


@pytest.fixture(scope="package")
def sphinx_gallery_config(sphinx_config):
    return sphinx_config["sphinx_gallery_conf"]


@pytest.fixture(scope="package")
def sphinx_gallery_scripts(sphinx_gallery_config):
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

    with utils.temp_add_to_sys_path(*dirs, root=None):
        yield tuple(scripts)


@pytest.fixture(scope="package", autouse=True)
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


@pytest.fixture(scope="package", autouse=True)
def patch_matplotlib_figures(package_mocker):
    package_mocker.patch(mocks.make_mock_target("image", "show_image"))
    package_mocker.patch(
        mocks.make_mock_target("pyplot", "new_figure_manager", pkg="matplotlib")
    )


@pytest.fixture(autouse=True)
def patch_optim_loops(mocker):
    def optim_loop_side_effect(input, *args, **kwargs):
        return input

    for name in (
        "default_image_optim_loop",
        "default_image_pyramid_optim_loop",
        "default_transformer_optim_loop",
        "default_transformer_epoch_optim_loop",
    ):
        mocker.patch(
            mocks.make_mock_target("optim", name), side_effect=optim_loop_side_effect
        )

    # Since the beginner example "NST without pystiche" does not use a builtin
    # optimization loop we are patching the optimizer. The actual computation happens
    # inside a closure. Thus, the loop will run, albeit with an almost empty body.
    mocker.patch(mocks.make_mock_target("optim", "LBFGS", pkg="torch"))
