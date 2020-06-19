from os import path

try:
    import yaml
    from pytorch_wheel_installer.core import find_links
except ImportError:
    msg = "Please install pyyaml and pytorch_wheel_selector prior to running this."
    raise RuntimeError(msg)


def extract_language_from_rtd_config(root, file=".readthedocs.yml"):
    with open(path.join(root, file)) as fh:
        data = yaml.load(fh, Loader=yaml.FullLoader)

    python_version = str(data["python"]["version"])
    return f"py{python_version.replace('.', '')}"


def main(
    root=".",
    distributions=("torch", "torchvision"),
    backend="cpu",
    language=None,
    platform="linux",
    file="requirements-rtd.txt",
):
    if language is None:
        language = extract_language_from_rtd_config(path.join(root, ".."))

    links = find_links(distributions, backend, language, platform)
    with open(path.join(root, file), "w") as fh:
        fh.write("\n".join(links) + "\n")


if __name__ == "__main__":
    main()
