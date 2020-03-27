from importlib import import_module
import os
from os import path
import re
import itertools
from setuptools import find_packages
import pystiche
from utils import PysticheTestCase


class TestCase(PysticheTestCase):
    def test_import(self):
        def find_modules(dir, package=None):
            if package is not None:
                dir = path.join(dir, package.replace(".", os.sep))
            files = os.listdir(dir)
            modules = []
            for file in files:
                name, ext = path.splitext(file)
                if ext == ".py" and not name.startswith("_"):
                    module = f"{package}." if package is not None else ""
                    module += name
                    modules.append(module)
            return modules

        public_packages = [
            package
            for package in find_packages(self.package_root)
            if not package.startswith("_")
        ]

        public_modules = find_modules(self.package_root)
        for package in public_packages:
            public_modules.extend(find_modules(self.package_root, package=package))

        for module in itertools.chain(public_packages, public_modules):
            import_module(f"pystiche.{module}")

    def test_about(self):
        for attr in (
            "name",
            "description",
            "version",
            "url",
            "license",
            "author",
            "author_email",
        ):
            self.assertIsInstance(getattr(pystiche, f"__{attr}__"), str)


if __name__ == "__main__":
    unittest.main()
