# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full list see
# the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Imports ---------------------------------------------------------------------------

import os
import re
import shutil
import warnings
from datetime import datetime
from distutils.util import strtobool
from importlib_metadata import metadata as extract_metadata
from os import path
from urllib.parse import urljoin

from sphinx_gallery.sorting import ExampleTitleSortKey, ExplicitOrder

import torch

from pystiche.misc import download_file

# -- Run config ------------------------------------------------------------------------


def get_bool_env_var(name, default=False):
    try:
        return bool(strtobool(os.environ[name]))
    except KeyError:
        return default


run_by_github_actions = get_bool_env_var("GITHUB_ACTIONS")
run_by_travis_ci = get_bool_env_var("TRAVIS")
run_by_appveyor = get_bool_env_var("APPVEYOR")
run_by_rtd = get_bool_env_var("READTHEDOCS")
run_by_ci = (
    run_by_github_actions
    or run_by_travis_ci
    or run_by_appveyor
    or run_by_rtd
    or get_bool_env_var("CI")
)

# -- Path setup ------------------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory, add
# these directories to sys.path here. If the directory is relative to the documentation
# root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))


# -- Project information ---------------------------------------------------------------

metadata = extract_metadata("pystiche")

project = metadata["name"]
author = metadata["author"]
copyright = f"{datetime.now().year}, {author}"
release = metadata["version"]
canonical_version = release.split("+")[0]
version = ".".join(canonical_version.split(".")[:3])


# -- General configuration -------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions coming
# with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    "sphinx_autodoc_typehints",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and directories to
# ignore when looking for source files. This pattern also affects html_static_path and
# html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- intersphinx configuration ---------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.6", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchvision": ("https://pytorch.org/docs/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/1.18/", None),
    "requests": ("https://requests.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org", None),
}


# -- sphinx-gallery configuration ------------------------------------------------------

plot_gallery = get_bool_env_var("PYSTICHE_PLOT_GALLERY", default=True) and not run_by_ci
download_gallery = get_bool_env_var("PYSTICHE_DOWNLOAD_GALLERY") or run_by_ci

if download_gallery:
    base = "https://download.pystiche.org/galleries/"
    is_dev = version != release
    file = "master.zip" if is_dev else f"v{version}.zip"

    url = urljoin(base, file)
    print(f"Downloading pre-built galleries from {url}")
    download_file(url, file)

    try:
        shutil.rmtree("galleries")
    except FileNotFoundError:
        pass
    shutil.unpack_archive(file, extract_dir=".")
    os.remove(file)

    # This is workaround for a bug in sphinx-gallery that replaces absolute with
    # relative paths. See https://github.com/pmeier/pystiche/pull/325 for details.
    index_file = path.join("galleries", "examples", "index.rst")
    with open(index_file, "r") as fh:
        content = fh.read()
    content = re.sub(
        r"(?P<file>examples_(python|jupyter)\.zip) <[\w/.]+>",
        r"\g<file> <\g<file>>",
        content,
    )
    with open(index_file, "w") as fh:
        fh.write(content)

    extensions.remove("sphinx_gallery.gen_gallery")
    extensions.append("sphinx_gallery.load_style")
    plot_gallery = False

if plot_gallery and not torch.cuda.is_available():
    msg = (
        "The galleries will be built, but CUDA is not available. "
        "This will take a long time."
    )
    print(msg)


def show_cuda_memory(func):
    torch.cuda.reset_peak_memory_stats()
    out = func()

    stats = torch.cuda.memory_stats()
    peak_bytes_usage = stats["allocated_bytes.all.peak"]
    memory = peak_bytes_usage / 1024 ** 2

    return memory, out


class PysticheExampleTitleSortKey(ExampleTitleSortKey):
    def __call__(self, filename):
        # The beginner example *without* pystiche is placed before the example *with*
        # to clarify the narrative.
        if filename == "example_nst_without_pystiche.py":
            return "1"
        elif filename == "example_nst_with_pystiche.py":
            return "2"
        else:
            return super().__call__(filename)


sphinx_gallery_conf = {
    "examples_dirs": path.join(PROJECT_ROOT, "examples"),
    "gallery_dirs": path.join("galleries", "examples"),
    "filename_pattern": os.sep + "example_",
    "line_numbers": True,
    "remove_config_comments": True,
    "plot_gallery": plot_gallery,
    "subsection_order": ExplicitOrder(
        [
            path.join("..", "..", "examples", sub_gallery)
            for sub_gallery in ("beginner", "advanced")
        ]
    ),
    "within_subsection_order": PysticheExampleTitleSortKey,
    "show_memory": show_cuda_memory if torch.cuda.is_available() else True,
}

# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=(
        "Matplotlib is currently using agg, which is a non-GUI backend, so cannot show "
        "the figure."
    ),
)


# -- Options for HTML output -----------------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for a list of
# builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here, relative
# to this directory. They are copied after the builtin static files, so a file named
# "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]


# -- Latex / Mathjax config ------------------------------------------------------------

with open("custom_cmds.tex", "r") as fh:
    custom_cmds = fh.read()

latex_elements = {"preamble": custom_cmds}

mathjax_inline = [r"\(" + custom_cmds, r"\)"]
mathjax_display = [r"\[" + custom_cmds, r"\]"]
