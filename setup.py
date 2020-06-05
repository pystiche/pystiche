import subprocess
from os import path
from setuptools import find_packages, setup

PROJECT_ROOT = path.abspath(path.dirname(__file__))
PACKAGE_NAME = "pystiche"
PACKAGE_ROOT = path.join(PROJECT_ROOT, PACKAGE_NAME)


about = {}
with open(path.join(PACKAGE_ROOT, "__about__.py"), "r") as fh:
    exec(fh.read(), about)

with open(path.join(PROJECT_ROOT, "README.rst"), "r") as fh:
    long_description = fh.read()


class Git:
    def run(self, *cmds, cwd=None):
        return subprocess.check_output(("git", *cmds), cwd=cwd).decode("utf-8").strip()

    def is_available(self) -> bool:
        try:
            self.run("--help")
            return True
        except subprocess.CalledProcessError:
            return False

    def is_repo(self, dir: str) -> bool:
        return path.exists(path.join(dir, ".git"))

    def is_dirty(self, dir: str) -> bool:
        return bool(self.run("status", "-uno", "--porcelain", cwd=dir))

    def hash(self, dir: str) -> str:
        return self.run("rev-parse", "--short", "HEAD", cwd=dir)


if about["__is_dev_version__"]:
    __version__ = f"{about['__base_version__']}+dev"

    git = Git()
    if git.is_available() and git.is_repo(PROJECT_ROOT):
        __version__ += f".{git.hash(PROJECT_ROOT)}"

        if git.is_dirty(PROJECT_ROOT):
            __version__ += ".dirty"

    with open(path.join(PACKAGE_ROOT, "__version__"), "w") as fh:
        fh.write(__version__)

    about["__version__"] = __version__

install_requires = (
    "torch>=1.4.0",
    "torchvision>=0.5.0",
    "pillow",
    "numpy",
    "requests",
    "typing_extensions",
)
test_requires = ("pytest", "pyimagetest", "pillow_affine", "dill", "pytest-subtests")

doc_requires = (
    "sphinx < 3.0.0",
    "sphinxcontrib-bibtex",
    "sphinx_autodoc_typehints",
    "sphinx-gallery>=0.7.0",
    # Install additional sphinx-gallery dependencies
    # https://sphinx-gallery.github.io/stable/index.html#install-via-pip
    "matplotlib",
    "sphinx_rtd_theme",
)

dev_requires = (
    *test_requires,
    *doc_requires,
    "isort",
    "black",
    "flake8",
    "mypy",
    "pre-commit",
    "pyyaml",
)

extras_require = {
    "test": test_requires,
    "doc": doc_requires,
    "dev": dev_requires,
}

classifiers = (
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
)

setup(
    name=about["__name__"],
    description=about["__description__"],
    version=about["__version__"],
    url=about["__url__"],
    license=about["__license__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_packages(where=PROJECT_ROOT, exclude=("docs", "examples", "tests")),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.6",
    classifiers=classifiers,
)
