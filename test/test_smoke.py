import os
from os import path
from setuptools import find_packages
from importlib import import_module
import itertools
import unittest
import pystiche


class Tester(unittest.TestCase):
    def test_import(self):
        here = path.abspath(path.dirname(__file__))
        pystiche_root = path.join(here, "..", "pystiche")

        packages = find_packages(pystiche_root)

        modules = []
        for package in packages:
            files = os.listdir(path.join(pystiche_root, package.replace(".", os.sep)))
            for file in files:
                name, ext = path.splitext(file)
                if ext == ".py" and name != "__init__":
                    modules.append(f"{package}.{name}")

        for module in itertools.chain(packages, modules):
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
