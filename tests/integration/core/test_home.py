import os
import tempfile
from os import path

import pystiche


def test_home_default():
    actual = pystiche.home()
    desired = path.expanduser(path.join("~", ".cache", "pystiche"))
    assert actual == desired


def test_home_env():
    tmp_dir = tempfile.mkdtemp()
    pystiche_home = os.getenv("PYSTICHE_HOME")
    os.environ["PYSTICHE_HOME"] = tmp_dir
    try:
        actual = pystiche.home()
        desired = tmp_dir
        assert actual == desired
    finally:
        if pystiche_home is None:
            del os.environ["PYSTICHE_HOME"]
        else:
            os.environ["PYSTICHE_HOME"] = pystiche_home
