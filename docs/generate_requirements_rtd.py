import configparser
import re
from os import path

try:
    import light_the_torch as ltt
    import yaml

    assert ltt.__version__ >= "0.2"
except (ImportError, AssertionError):
    msg = "Please install pyyaml and light-the-torch>=0.2 prior to running this."
    raise RuntimeError(msg)


DEPS_SUBSTITUTION_PATTERN = re.compile(r"\{\[(?P<section>[a-zA-Z\-]+)\]deps\}")


def main(
    root=".", file=path.join("docs", "requirements-rtd.txt"),
):
    python_version = extract_python_version_from_rtd_config(root)

    deps = extract_docs_deps_from_tox_config(root)
    deps.extend(find_pytorch_wheel_links(root, python_version))

    with open(file, "w") as fh:
        fh.write("\n".join(deps) + "\n")


def extract_python_version_from_rtd_config(root, file=".readthedocs.yml"):
    with open(path.join(root, file)) as fh:
        data = yaml.load(fh, Loader=yaml.FullLoader)

    return str(data["python"]["version"])


def extract_docs_deps_from_tox_config(root, file="tox.ini", section="docs-common"):
    config = configparser.ConfigParser()
    config.read(path.join(root, file))

    deps = []
    sections = [section]
    for section in sections:
        for dep in config[section]["deps"].strip().split("\n"):
            match = DEPS_SUBSTITUTION_PATTERN.match(dep)
            if match is None:
                deps.append(dep)
            else:
                sections.append(match.group("section"))
    return deps


def find_pytorch_wheel_links(
    root, python_version, computation_backend="cpu", platform="linux_x86_64",
):
    return ltt.find_links(
        [root],
        computation_backend=computation_backend,
        python_version=python_version,
        platform=platform,
    )


if __name__ == "__main__":
    project_root = path.abspath(path.join(path.dirname(__file__), ".."))
    main(project_root)
