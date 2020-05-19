from importlib.util import module_from_spec, spec_from_file_location
from os import path
from setuptools import find_packages, setup

PROJECT_ROOT = path.abspath(path.dirname(__file__))
PACKAGE_NAME = "pystiche"
PACKAGE_ROOT = path.join(PROJECT_ROOT, PACKAGE_NAME)


def load_git_module():
    spec = spec_from_file_location(PACKAGE_NAME, path.join(PACKAGE_ROOT, "_git.py"),)
    git = module_from_spec(spec)
    spec.loader.exec_module(git)
    return git


git = load_git_module()
about = {"git": git, "_PROJECT_ROOT": PROJECT_ROOT}
with open(path.join(PACKAGE_ROOT, "__about__.py"), "r") as fh:
    exec(fh.read(), about)

with open(path.join(PROJECT_ROOT, "README.rst"), "r") as fh:
    long_description = fh.read()

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
    "sphinx",
    "sphinx_autodoc_typehints",
    "sphinxcontrib-bibtex",
    "sphinx_rtd_theme",
    "sphinx-gallery",
    # Install additional sphinx-gallery dependencies
    # https://sphinx-gallery.github.io/stable/index.html#install-via-pip
    "matplotlib",
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
