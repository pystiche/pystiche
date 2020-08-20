import builtins
import os
import sys
import unittest.mock
from distutils import dir_util

import pystiche

__all__ = [
    "make_mock_target",
    "patch_imports",
    "ContextMock",
    "patch_models_load_state_dict_from_url",
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


class ContextMock:
    def __init__(self, mock):
        self.__mock__ = mock

    def __getattribute__(self, item):
        if item == "__mock__":
            return object.__getattribute__(self, "__mock__")
        else:
            return getattr(self.__mock__, item)

    def __setattr__(self, key, value):
        if key == "__mock__":
            object.__setattr__(self, key, value)
        else:
            self.__mock__.__setattr__(key, value)

    def __call__(self, *args, **kwargs):
        return self.__mock__(*args, **kwargs)

    def __enter__(self):
        self.__mock__.reset_mock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def patch_models_load_state_dict_from_url(mocker=DEFAULT_MOCKER):
    return ContextMock(
        mocker.patch(
            make_mock_target(
                "enc",
                "models",
                "utils",
                "ModelMultiLayerEncoder",
                "load_state_dict_from_url",
            )
        )
    )


def patch_home(home, copy=True, mocker=DEFAULT_MOCKER):
    if copy:
        dir_util.copy_tree(pystiche.home(), home)

    return mocker.patch.dict(os.environ, values={"PYSTICHE_HOME": home})
