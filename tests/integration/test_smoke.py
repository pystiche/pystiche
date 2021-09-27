import builtins
import importlib
import os
import pathlib
import re
import sys

import pytest

PACKAGE_NAME = "pystiche"
PROJECT_ROOT = pathlib.Path(__file__).parents[2]
PACKAGE_ROOT = PROJECT_ROOT / PACKAGE_NAME


def collect_modules():
    def is_private(path):
        return pathlib.Path(path).name.startswith("_")

    def path_to_module(path):
        return str(pathlib.Path(path).with_suffix("")).replace(os.sep, ".")

    modules = []
    for root, dirs, files in os.walk(PACKAGE_ROOT):
        if is_private(root) or "__init__.py" not in files:
            del dirs[:]
            continue

        path = pathlib.Path(root).relative_to(PROJECT_ROOT)
        modules.append(path_to_module(path))

        for file in files:
            if is_private(file) or not file.endswith(".py"):
                continue

            modules.append(path_to_module(path / file))

    return modules


@pytest.mark.parametrize("module", collect_modules())
def test_importability(module):
    importlib.import_module(module)


def import_package_under_test():
    try:
        return importlib.import_module(PACKAGE_NAME)
    except Exception as error:
        raise RuntimeError(
            f"The package '{PACKAGE_NAME}' could not be imported. "
            f"Check the results of tests/test_smoke.py::test_importability for details."
        ) from error


def test_version_installed():
    def is_canonical(version):
        # Copied from
        # https://www.python.org/dev/peps/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
        match = re.match(
            r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$",
            version,
        )
        return match is not None

    def is_dev(version):
        match = re.search(r"\+g[\da-f]{7}([.]\d{14})?", version)
        if match is not None:
            return is_canonical(version[: match.span()[0]])
        else:
            return False

    put = import_package_under_test()
    assert is_canonical(put.__version__) or is_dev(put.__version__)


def patch_imports(
    mocker, *names, retain_condition=None, import_error_condition=None,
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
            raise ImportError()

        return __import__(name, globals, locals, fromlist, level)

    mocker.patch.object(builtins, "__import__", new=patched_import)

    values = {
        name: module for name, module in sys.modules.items() if retain_condition(name)
    }
    mocker.patch.dict(sys.modules, clear=True, values=values)


def test_version_not_installed(mocker):
    def import_error_condition(name, globals, locals, fromlist, level):
        return name == "_version" and fromlist == ("version",)

    patch_imports(mocker, PACKAGE_NAME, import_error_condition=import_error_condition)

    put = import_package_under_test()
    assert put.__version__ == "UNKNOWN"
