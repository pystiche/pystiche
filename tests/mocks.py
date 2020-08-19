import builtins
import os
import sys
import unittest.mock
from distutils import dir_util

import pystiche

__all__ = [
    "make_mock_target",
    "patch_imports",
    "patch_multi_layer_encoder_load_weights",
    "patch_home",
]

DEFAULT_MOCKER = unittest.mock


def make_mock_target(*args, pkg="pystiche"):
    return ".".join((pkg, *args))


def patch_imports(
    names,
    clear=True,
    retain_condition=None,
    import_error_condition=None,
    mocker=DEFAULT_MOCKER,
):
    if retain_condition is None:

        def retain_condition(name):
            return not any(name.startswith(name_) for name_ in names)

    if import_error_condition is None:

        def import_error_condition(name, globals, locals, fromlist, level):
            direct = name in names
            indirect = fromlist is not None and any(
                from_ in names for from_ in fromlist
            )
            return direct or indirect

    __import__ = builtins.__import__

    def patched_import(name, globals, locals, fromlist, level):
        if import_error_condition(name, globals, locals, fromlist, level):
            raise ImportError

        return __import__(name, globals, locals, fromlist, level)

    mocker.patch.object(builtins, "__import__", new=patched_import)
    if clear:
        values = {
            name: module
            for name, module in sys.modules.items()
            if retain_condition(name)
        }
    else:
        values = {}
    mocker.patch.dict(
        sys.modules, clear=clear, values=values,
    )


_MULTI_LAYER_ENCODER_LOAD_WEIGHTS_TARGETS = {
    "vgg": make_mock_target(
        "enc", "models", "vgg", "VGGMultiLayerEncoder", "_load_weights"
    ),
    "alexnet": make_mock_target(
        "enc", "models", "alexnet", "AlexNetMultiLayerEncoder", "_load_weights",
    ),
}


class MockDict(dict):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for mock in self.values():
            mock.reset_mock()


def patch_multi_layer_encoder_load_weights(models=None, mocker=DEFAULT_MOCKER):
    if models is None:
        models = _MULTI_LAYER_ENCODER_LOAD_WEIGHTS_TARGETS.keys()

    return MockDict(
        (
            (model, mocker.patch(_MULTI_LAYER_ENCODER_LOAD_WEIGHTS_TARGETS[model]))
            for model in models
        )
    )


def patch_home(home, copy=True, mocker=DEFAULT_MOCKER):
    if copy:
        dir_util.copy_tree(pystiche.home(), home)

    return mocker.patch.dict(os.environ, values={"PYSTICHE_HOME": home})
