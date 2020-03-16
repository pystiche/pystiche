from os import path
from setuptools import setup, find_packages

about = {}
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "pystiche", "__about__.py"), "r") as fh:
    exec(fh.read(), about)

with open(path.join(here, "README.md"), "r") as fh:
    long_description = fh.read()

install_requires = ("torch>=1.4.0", "torchvision>=0.5.0", "pillow", "numpy", "requests")

extras_require = {
    # FIXME: move to a released versions
    "dev": (
        "pre-commit",
        "mypy",
        "pyimagetest@https://github.com/pmeier/pyimagetest/archive/master.zip",
        "pillow_affine@https://github.com/pmeier/pillow_affine/archive/master.zip",
    )
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
    packages=find_packages(exclude=("test",)),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.6",
    classifiers=classifiers,
)
