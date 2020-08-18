import importlib
import pkgutil
import re

from tests.mocks import patch_imports


def import_package_under_test():
    import pystiche as package_under_test

    return package_under_test


put = import_package_under_test()


def test_importability(subtests):
    def is_private(name):
        return name.rsplit(".", 1)[-1].startswith("_")

    def onerror(name):
        if is_private(name):
            return

        with subtests.test(name=name):
            raise

    for finder, name, is_package in pkgutil.walk_packages(
        path=put.__path__, prefix=f"{put.__name__}.", onerror=onerror,
    ):
        if is_private(name):
            continue

        if not is_package:
            try:
                importlib.import_module(name)
            except Exception:
                onerror(name)


def test_version_installed():
    def is_canonical(version):
        # Copied from
        # https://www.python.org/dev/peps/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
        return (
            re.match(
                r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$",
                version,
            )
            is not None
        )

    def is_dev(version):
        match = re.search(r"\+g[\da-f]{7}([.]\d{14})?", version)
        if match is not None:
            return is_canonical(version[: match.span()[0]])
        else:
            return False

    assert is_canonical(put.__version__) or is_dev(put.__version__)


def test_version_not_installed(mocker):
    def import_error_condition(name, globals, locals, fromlist, level):
        return name == "_version" and fromlist == ("version",)

    patch_imports(
        (put.__name__,), import_error_condition=import_error_condition, mocker=mocker
    )

    reimported_put = import_package_under_test()
    assert reimported_put.__version__ == "UNKNOWN"
