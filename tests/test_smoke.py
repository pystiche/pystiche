import contextlib
import itertools
import os
import re
import shutil
import tempfile
import unittest
from importlib import import_module, util
from os import path

from setuptools import find_packages

PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), ".."))
PACKAGE_NAME = "pystiche"
PACKAGE_ROOT = path.join(PROJECT_ROOT, PACKAGE_NAME)


def load_module(location):
    name, ext = path.splitext(path.basename(location))
    if ext != ".py":
        location = path.join(location, "__init__.py")

    spec = util.spec_from_file_location(name, location=location)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package_under_test = load_module(PACKAGE_ROOT)
git = load_module(path.join(PACKAGE_ROOT, "_git.py"))

skip_if_git_not_available = unittest.skipIf(
    not git.is_available(), "git is not available."
)


class TestSmoke(unittest.TestCase):
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
            for package in find_packages(PACKAGE_ROOT)
            if not package.startswith("_")
        ]

        public_modules = find_modules(PACKAGE_ROOT)
        for package in public_packages:
            public_modules.extend(find_modules(PACKAGE_ROOT, package=package))

        for module in itertools.chain(public_packages, public_modules):
            import_module(f".{module}", package=PACKAGE_NAME)

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
            self.assertIsInstance(getattr(package_under_test, f"__{attr}__"), str)

    def test_name(self):
        self.assertEqual(package_under_test.__name__, PACKAGE_NAME)

    def test_version(self):
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
            match = re.search(r"\+dev([.][\da-f]{7}([.]dirty)?)?$", version)
            if match is not None:
                return is_canonical(version[: match.span()[0]])
            else:
                return False

        version = package_under_test.__version__
        self.assertTrue(is_canonical(version) or is_dev(version))


class TestGit(unittest.TestCase):
    @staticmethod
    @contextlib.contextmanager
    def get_tmp_git_dir(**mkdtemp_kwargs):
        tmp_git_dir = tempfile.mkdtemp(**mkdtemp_kwargs)
        try:
            if git.is_available():
                git.run("init", cwd=tmp_git_dir)
            else:
                os.mkdir(path.join(tmp_git_dir, ".git"))
            yield tmp_git_dir
        finally:
            shutil.rmtree(tmp_git_dir)

    def test_git_is_available_smoke(self):
        self.assertIsInstance(git.is_available(), bool)

    def test_git_is_repo(self):
        with self.get_tmp_git_dir() as repo:
            self.assertTrue(git.is_repo(repo))

    @skip_if_git_not_available
    def test_git_is_dirty(self):
        with self.get_tmp_git_dir() as repo:
            with open(path.join(repo, "dirty"), "wb"):
                pass

            self.assertFalse(git.is_dirty(repo))

            git.run("add", "dirty", cwd=repo)

            self.assertTrue(git.is_dirty(repo))

    @skip_if_git_not_available
    def test_git_hash(self):
        with self.get_tmp_git_dir() as repo:
            with open(path.join(repo, "dirty"), "wb"):
                pass
            git.run("add", "dirty", cwd=repo)
            git.run("config", "user.name", "'pystiche test suite'", cwd=repo)
            git.run("config", "user.email", "'testsuite@pystiche.org'", cwd=repo)
            git.run("commit", "-m", "'test commit'", cwd=repo)

            self.assertIsNotNone(re.match(r"^[0-9a-f]{7}$", git.hash(repo)))
