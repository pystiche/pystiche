import contextlib
import functools
import itertools
import os
import re
import shutil
import stat
import sys
import tempfile
import warnings
from typing import Type
from distutils import dir_util
from os import path

import igittigitt
import pytest

import torch

__all__ = [
    "get_tmp_dir",
    "skip_if_cuda_not_available",
    "watch_dir",
    "temp_add_to_sys_path",
]


PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), ".."))


# Adapted from
# https://pypi.org/project/pathutils/
def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    if not os.access(path, os.W_OK):
        # Is the error an access error ?
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


@contextlib.contextmanager
def get_tmp_dir(**mkdtemp_kwargs):
    tmp_dir = tempfile.mkdtemp(**mkdtemp_kwargs)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, onerror=onerror)


skip_if_cuda_not_available = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available."
)


def walk_git_project(*args, exlude_git_dir=True, **kwargs):
    parser = igittigitt.IgnoreParser()
    parser.parse_rule_files(PROJECT_ROOT)
    is_ignored = parser.match
    for root, dirs, files in os.walk(*args, **kwargs):

        if is_ignored(root):
            continue

        ignored_dirs = [
            dir for dir in dirs if is_ignored(dir) or (exlude_git_dir and dir == ".git")
        ]
        # remove them in-place to avoid further exploration
        for dir in ignored_dirs:
            dirs.remove(dir)

        files = [file for file in files if not is_ignored(file)]
        yield root, dirs, files


def _find_dirs_and_files(top, honor_gitignore=True):
    walk = walk_git_project if honor_gitignore else os.walk

    dirs = set()
    files = set()
    for root, rel_dirs, rel_files in walk(top):
        dirs.update(path.join(root, rel_dir) for rel_dir in rel_dirs)
        files.update(path.join(root, rel_file) for rel_file in rel_files)
    return dirs, files


def _rel_to_abs_path(rel_path, root):
    if root is None:
        return rel_path

    return path.join(root, rel_path)


@contextlib.contextmanager
def watch_dir(
    rel_path,
    root=PROJECT_ROOT,
    error_on_diff=False,
    remove_diff=None,
    honor_gitignore=True,
):
    abs_path = _rel_to_abs_path(rel_path, root)

    if remove_diff is None:
        remove_diff = not error_on_diff

    old_dirs, old_files = _find_dirs_and_files(
        abs_path, honor_gitignore=honor_gitignore
    )
    try:
        yield
    finally:
        new_dirs, new_files = _find_dirs_and_files(
            abs_path, honor_gitignore=honor_gitignore
        )

        diff_dirs = new_dirs - old_dirs
        diff_files = new_files - old_files

        if not (diff_dirs or diff_files):
            return

        if remove_diff:
            for dir in diff_dirs:
                dir_util.remove_tree(dir)

            for file in diff_files:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    # file might have been already removed with dir removal
                    pass

        if error_on_diff:
            msg = "The following directories and files were added:\n\n" "\n".join(
                itertools.chain(diff_dirs, diff_files)
            )
            raise pytest.UsageError(msg)


@contextlib.contextmanager
def temp_add_to_sys_path(*rel_paths, root=PROJECT_ROOT):
    abs_paths = [_rel_to_abs_path(rel_path, root) for rel_path in rel_paths]
    for abs_path in abs_paths:
        sys.path.insert(0, abs_path)
    try:
        yield
    finally:
        for abs_path in abs_paths:
            sys.path.remove(abs_path)


def maybe_update(dct, key, val):
    if key not in dct:
        dct[key] = val


def exec_file(rel_path, root=PROJECT_ROOT, globals=None, locals=None):
    abs_path = _rel_to_abs_path(rel_path, root)
    if globals is None:
        globals = {}
    maybe_update(globals, "__file__", abs_path)
    maybe_update(globals, "__name__", "__main__")

    with open(abs_path, "r") as fh:
        exec(fh.read(), globals, locals)

    return globals, locals


FUNCTION_NAME_PATTERN = re.compile(r"<function (?P<name>\w+) at 0x[0-9a-e]+")


def extract_fn_name(fn):
    match = FUNCTION_NAME_PATTERN.match(str(fn))
    if not match:
        raise RuntimeError
    return match.group("name")


def suppress_deprecation_warning(*categories: Type[Warning]):
    if not categories:
        categories = (UserWarning,)
    def deprecation_decor(test_fn):
        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            for category in categories:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=category, module=r'.*pystiche')
                    return test_fn(*args, **kwargs)
        return wrapper
    return deprecation_decor
