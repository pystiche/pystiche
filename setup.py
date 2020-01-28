from setuptools import setup, find_packages
import pystiche

version = pystiche.__version__

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ("numpy", "torch>=1.4.0", "pillow", "torchvision>=0.5.0")

extras_require = {
    # FIXME: move to a released version
    "testing": ("pyimagetest@https://github.com/pmeier/pyimagetest/archive/master.zip",)
}

classifiers = (
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
)

setup(
    name="pystiche",
    description="pystiche is a framework for Neural Style Transfer (NST) algorithms built upon PyTorch",
    version=version,
    url="https://github.com/pmeier/pystiche",
    license="BSD-3",
    author="Philip Meier",
    author_email="github.pmeier@posteo.de",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("test",)),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.6",
    classifiers=classifiers,
)
