from setuptools import setup, find_packages

requirements = (
    "numpy",
    "torch >= 1.2.0",
    "torchvision >= 0.4.0",
    "matplotlib",
    "requests",
    "django",
)

setup(
    name="pystiche",
    version="0.1",
    author="Philip Meier",
    author_email="pystiche@posteo.de",
    url="https://github.com/pmeier/pystiche",
    description="The pystiche project is a free, open-source framework for Neural Style Transfer (NST) algorithms",
    license="BSD-3",
    packages=find_packages(exclude=("images", "replication", "test", "webapp")),
    install_requires=requirements,
)
