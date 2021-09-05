import os
import pathlib
import re
import subprocess
import sys

# check if this can be unified into one pattern
WARNING_PATTERN = r"WARNING: (?P<warning>.*?)$"
FILE_PATTERN = re.compile(r"(?P<location>.*?): " + WARNING_PATTERN)
DOCSTRING_PATTERN = re.compile(r".*?(?P<location>docstring of .*?): " + WARNING_PATTERN)

# take regular expressions
IGNORED_WARNINGS = {
    "py:class reference target not found: torchvision.models.VGG",
    "py:class reference target not found: torchvision.models.AlexNet",
}


HERE = pathlib.Path(__file__).parent
DOCS_DIR = HERE.parent / "docs"


def main():
    env = os.environ.copy()
    env.setdefault("PYSTICHE_PLOT_GALLERY", str(False))

    # drop -a after build is redirected
    output = subprocess.check_output(
        (
            "sphinx-build",
            "-anqb",
            "html",
            str(DOCS_DIR / "source"),
            str(DOCS_DIR / "build"),
        ),
        env=env,
        stderr=subprocess.STDOUT,
    )

    atleast_one = False
    for line in output.decode().strip().splitlines():
        pattern = DOCSTRING_PATTERN if "docstring of" in line else FILE_PATTERN

        match = pattern.match(line)
        if match is None:
            raise RuntimeError

        location = match.group("location")
        warning = match.group("warning")

        if warning in IGNORED_WARNINGS:
            continue
        atleast_one = True

        print(f"{location}: {warning}", file=sys.stderr)

    sys.exit(int(atleast_one))


if __name__ == "__main__":
    # take source dir and put build into tmp
    main()
