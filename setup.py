from setuptools import setup, find_packages

requirements = (
    "numpy",
    "torch >= 1.2.0",
    "pillow <= 6.2.0",
    "torchvision >= 0.4.0",
)

testing_requires = (
    "pyimagetest@https://github.com/pmeier/pyimagetest/archive/master.zip",
)

setup(
    name="pystiche",
    version="0.2dev",
    author="Philip Meier",
    author_email="github.pmeier@posteo.de",
    url="https://github.com/pmeier/pystiche",
    description="pystiche is a framework for Neural Style Transfer (NST) algorithms built upon PyTorch",
    license="BSD-3",
    packages=find_packages(exclude=("test",)),
    install_requires=requirements,
    extras_require={"testing": testing_requires},
)
