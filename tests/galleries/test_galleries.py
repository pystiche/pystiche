import pytest


@pytest.mark.slow
def test_gallery_scripts_smoke(subtests, sphinx_gallery_scripts):
    for script in sphinx_gallery_scripts:
        with subtests.test(script):
            __import__(script)
