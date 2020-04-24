# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Imports -----------------------------------------------------------------

import os
import warnings
from datetime import datetime
from distutils.util import strtobool as _strtobool
from os import path


def strtobool(val):
    return bool(_strtobool(val))


# -- Run config --------------------------------------------------------------


def get_bool_env_var(name):
    try:
        return strtobool(os.environ[name])
    except KeyError:
        return False


run_by_travis_ci = get_bool_env_var("TRAVIS")
run_by_rtd = get_bool_env_var("READTHEDOCS")
exclude_gallery = (
    get_bool_env_var("PYSTICHE_EXCLUDE_GALLERY") or run_by_travis_ci or run_by_rtd
)


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))


# -- Project information -----------------------------------------------------

pkg_name = "pystiche"

about = {"_PROJECT_ROOT": PROJECT_ROOT}
with open(path.join(PROJECT_ROOT, pkg_name, "__about__.py"), "r") as fh:
    exec(fh.read(), about)

project = about["__name__"]
copyright = f"2019 - {datetime.now().year}, {about['__author__']}"
author = about["__author__"]
release = about["__version__"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Sphinx gallery configuration --------------------------------------------

if not exclude_gallery:
    extensions.append("sphinx_gallery.gen_gallery")

    sphinx_gallery_conf = {
        "examples_dirs": path.join(PROJECT_ROOT, "examples"),
        "gallery_dirs": "auto_examples",
        "filename_pattern": os.sep + "example_",
    }
else:
    warnings.warn("Sphinx gallery is excluded and will not be built!", UserWarning)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
