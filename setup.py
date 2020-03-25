from os import path
from setuptools import setup, find_packages

about = {}
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "pystiche", "__about__.py"), "r") as fh:
    exec(fh.read(), about)

with open(path.join(here, "README.md"), "r") as fh:
    long_description = fh.read()

install_requires = ("torch>=1.4.0", "torchvision>=0.5.0", "pillow", "numpy", "requests")

test_requires = ("pytest", "pyimagetest", "pillow_affine")

dev_requires = (*test_requires, "pre-commit")

extras_require = {
    "test": test_requires,
    "dev": dev_requires,
}

classifiers = (
    "Development Status :: 3 - Beta",
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
    packages=find_packages(where=here, exclude=("test",)),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.6",
    classifiers=classifiers,
)
