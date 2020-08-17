import os
import unittest.mock
from distutils import dir_util

import pystiche

__all__ = [
    "make_mock_target",
    "patch_multi_layer_encoder_load_weights",
    "patch_home",
]

DEFAULT_MOCKER = unittest.mock


def make_mock_target(*args, pkg="pystiche"):
    return ".".join((pkg, *args))


_MULTI_LAYER_ENCODER_LOAD_WEIGHTS_TARGETS = {
    "vgg": make_mock_target(
        "enc", "models", "vgg", "VGGMultiLayerEncoder", "_load_weights"
    ),
    "alexnet": make_mock_target(
        "enc", "models", "alexnet", "AlexNetMultiLayerEncoder", "_load_weights",
    ),
}


def patch_multi_layer_encoder_load_weights(models=None, mocker=DEFAULT_MOCKER):
    if models is None:
        models = _MULTI_LAYER_ENCODER_LOAD_WEIGHTS_TARGETS.keys()

    return {
        model: mocker.patch(_MULTI_LAYER_ENCODER_LOAD_WEIGHTS_TARGETS[model])
        for model in models
    }


def patch_home(home, copy=True, mocker=DEFAULT_MOCKER):
    if copy:
        dir_util.copy_tree(pystiche.home(), home)

    return mocker.patch.dict(os.environ, values={"PYSTICHE_HOME": home})
