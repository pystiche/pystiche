from os import path
from setuptools import setup, find_packages

about = {}
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "pystiche", "__about__.py"), "r") as fh:
    exec(fh.read(), about)

with open(path.join(here, "README.md"), "r") as fh:
    long_description = fh.read()

install_requires = ("torch>=1.4.0", "torchvision>=0.5.0", "Pillow", "numpy", "requests")

test_requires = ("pytest", "pyimagetest", "pillow_affine")

doc_requires = (
    "sphinx",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinx-gallery",
    # Install additional sphinx-gallery dependencies
    # https://sphinx-gallery.github.io/stable/index.html#install-via-pip
    "matplotlib",
)

dev_requires = (*test_requires, *doc_requires, "pre-commit", "BeautifulSoup4")

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
    long_description_content_type="text/markdown",
    packages=find_packages(where=here, exclude=("demo", "docs", "test", "tutorials")),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.6",
    classifiers=classifiers,
)
