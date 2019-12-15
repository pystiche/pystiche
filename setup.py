from setuptools import setup, find_packages

requirements = (
    "numpy",
    "torch >= 1.2.0",
    "torchvision >= 0.4.0",
)

setup(
    name="pystiche",
    version="0.2dev",
    author="Philip Meier",
    author_email="github.pmeier@posteo.de",
    url="https://github.com/pmeier/pystiche",
    description="pystiche project is a framework for Neural Style Transfer (NST) algorithms built upon PyTorch",
    license="BSD-3",
    packages=find_packages(exclude=("test",)),
    install_requires=requirements,
)
