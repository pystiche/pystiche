import contextlib
import os
import re
import shutil
from datetime import datetime
from distutils.util import strtobool
from importlib_metadata import metadata as extract_metadata
from os import path
from urllib.parse import urljoin

from sphinx_gallery.sorting import ExampleTitleSortKey, ExplicitOrder

import torch

from pystiche.misc import download_file

HERE = path.dirname(__file__)
PROJECT_ROOT = path.abspath(path.join(HERE, "..", ".."))


def get_bool_env_var(name, default=False):
    try:
        return bool(strtobool(os.environ[name]))
    except KeyError:
        return default


GITHUB_ACTIONS = get_bool_env_var("GITHUB_ACTIONS")
RTD = get_bool_env_var("READTHEDOCS")
CI = GITHUB_ACTIONS or RTD or get_bool_env_var("CI")


def project():
    extension = None

    metadata = extract_metadata("pystiche")
    project = metadata["name"]
    author = metadata["author"]
    copyright = f"{datetime.now().year}, {author}"
    release = metadata["version"]
    canonical_version = release.split("+")[0]
    version = ".".join(canonical_version.split(".")[:3])
    config = dict(
        project=project,
        author=author,
        copyright=copyright,
        release=release,
        version=version,
    )

    return extension, config


def autodoc():
    extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.napoleon",
        "sphinx_autodoc_typehints",
    ]

    config = None

    return extensions, config


def intersphinx():
    extension = "sphinx.ext.intersphinx"
    config = dict(
        intersphinx_mapping={
            "python": ("https://docs.python.org/3.6", None),
            "torch": ("https://pytorch.org/docs/stable/", None),
            "torchvision": ("https://pytorch.org/docs/stable/", None),
            "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
            "numpy": ("https://numpy.org/doc/1.18/", None),
            "requests": ("https://requests.readthedocs.io/en/stable/", None),
            "matplotlib": ("https://matplotlib.org", None),
        }
    )
    return extension, config


def html():
    extension = None

    config = dict(html_theme="sphinx_rtd_theme")

    return extension, config


def latex():
    extension = None

    with open(path.join(HERE, "custom_cmds.tex"), "r") as fh:
        custom_cmds = fh.read()
    config = dict(
        latex_elements={"preamble": custom_cmds},
        mathjax_inline=[r"\(" + custom_cmds, r"\)"],
        mathjax_display=[r"\[" + custom_cmds, r"\]"],
    )

    return extension, config


def bibtex():
    extension = "sphinxcontrib.bibtex"

    config = None

    return extension, config


def doctest():
    extension = "sphinx.ext.doctest"

    doctest_global_setup = """
import torch
from torch import nn

from pystiche import enc, ops, loss
import pystiche.ops.functional as F

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from unittest import mock

patcher = mock.patch(
    "pystiche.enc.models.utils.ModelMultiLayerEncoder.load_state_dict_from_url"
)
patcher.start()
"""

    doctest_global_cleanup = """
mock.patch.stopall()
"""
    config = dict(
        doctest_global_setup=doctest_global_setup,
        doctest_global_cleanup=doctest_global_cleanup,
    )

    return extension, config


def sphinx_gallery():
    extension = "sphinx_gallery.gen_gallery"

    plot_gallery = get_bool_env_var("PYSTICHE_PLOT_GALLERY", default=not CI)
    download_gallery = get_bool_env_var("PYSTICHE_DOWNLOAD_GALLERY", default=CI)

    def download():
        nonlocal extension
        nonlocal plot_gallery

        # version and release are available as soon as the project config is loaded
        version = globals()["version"]
        release = globals()["release"]

        base = "https://download.pystiche.org/galleries/"
        is_dev = version != release
        file = "master.zip" if is_dev else f"v{version}.zip"

        url = urljoin(base, file)
        print(f"Downloading pre-built galleries from {url}")
        download_file(url, file)

        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(path.join(HERE, "galleries"))
        shutil.unpack_archive(file, extract_dir=".")
        os.remove(file)

        extension = "sphinx_gallery.load_style"
        plot_gallery = False

    def show_cuda_memory(func):
        torch.cuda.reset_peak_memory_stats()
        out = func()

        stats = torch.cuda.memory_stats()
        peak_bytes_usage = stats["allocated_bytes.all.peak"]
        memory = peak_bytes_usage / 1024 ** 2

        return memory, out

    class PysticheExampleTitleSortKey(ExampleTitleSortKey):
        def __call__(self, filename):
            # The beginner example *without* pystiche is placed before the example
            # *with* to clarify the narrative.
            if filename == "example_nst_without_pystiche.py":
                return "1"
            elif filename == "example_nst_with_pystiche.py":
                return "2"
            else:
                return super().__call__(filename)

    if download_gallery:
        download()

    if plot_gallery and not torch.cuda.is_available():
        msg = (
            "The galleries will be built, but CUDA is not available. "
            "This will take a long time."
        )
        print(msg)

    sphinx_gallery_conf = {
        "examples_dirs": path.join(PROJECT_ROOT, "examples"),
        "gallery_dirs": path.join("galleries", "examples"),
        "filename_pattern": re.escape(os.sep) + r"example_\w+[.]py$",
        "ignore_pattern": re.escape(os.sep) + r"_\w+[.]py$",
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

    config = dict(sphinx_gallery_conf=sphinx_gallery_conf)

    return extension, config


extensions = []
for loader in (
    project,
    autodoc,
    intersphinx,
    html,
    latex,
    bibtex,
    doctest,
    sphinx_gallery,
):
    extension, config = loader()

    if extension:
        if isinstance(extension, str):
            extension = (extension,)
        extensions.extend(extension)

    if config:
        globals().update(config)
