import contextlib
from os import path
import tempfile
import shutil
import unittest

__all__ = ["PysticheTestCase", "get_tmp_dir"]


class PysticheTestCase(unittest.TestCase):
    project_root = path.abspath(path.join(path.dirname(__file__), ".."))

    @property
    def package_name(self):
        return "pystiche"

    @property
    def package_root(self):
        return path.join(self.project_root, self.package_name)

    @property
    def test_root(self):
        return path.join(self.project_root, "test")


@contextlib.contextmanager
def get_tmp_dir(**mkdtemp_kwargs):
    tmp_dir = tempfile.mkdtemp(**mkdtemp_kwargs)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)
