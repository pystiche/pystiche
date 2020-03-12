import unittest
from os import path
from setuptools import find_packages
from importlib import import_module
import pystiche


class Tester(unittest.TestCase):
    def test_import(self):
        here = path.abspath(path.dirname(__file__))
        for module in find_packages(path.join(here, "..", "pystiche")):
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
