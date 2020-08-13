import importlib
import pkgutil

import pystiche as package_under_test


def test_importability(subtests):
    def is_private(name):
        return name.rsplit(".", 1)[-1].startswith("_")

    def onerror(name):
        if is_private(name):
            return

        with subtests.test(name=name):
            raise

    for finder, name, is_package in pkgutil.walk_packages(
        path=package_under_test.__path__,
        prefix=f"{package_under_test.__name__}.",
        onerror=onerror,
    ):
        if is_private(name):
            continue

        if not is_package:
            try:
                importlib.import_module(name)
            except Exception:
                onerror(name)


def test_version_availability():
    assert isinstance(package_under_test.__version__, str)
