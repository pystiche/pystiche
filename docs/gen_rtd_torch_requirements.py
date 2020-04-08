from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from os import path

import yaml

PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), ".."))


def load_generator_main():
    name = "gen_torch_cpu_requirements"
    spec = spec_from_file_location(name, path.join(PROJECT_ROOT, f"{name}.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def parse_rtd_config(file=".readthedocs.yml"):
    def extract_file(data):
        python_install_data = data["python"]["install"]
        requirement_files = [
            option["requirements"]
            for option in python_install_data
            if "requirements" in option
        ]
        if len(requirement_files) > 1:
            raise RuntimeError

        return path.join(PROJECT_ROOT, requirement_files[0])

    def extract_language(data):
        python_data = data["python"]
        python_version = str(python_data["version"])
        return f"cp{python_version.replace('.', '')}"

    with open(path.join(PROJECT_ROOT, file)) as fh:
        data = yaml.load(fh, Loader=yaml.FullLoader)

    file = extract_file(data)
    language = extract_language(data)

    return file, language


def get_args():
    file, language = parse_rtd_config()
    return Namespace(file=file, language=language, abi=None, platform="linux_x86_64",)


def main():
    generator_main = load_generator_main()
    args = get_args()
    generator_main(args)


if __name__ == "__main__":
    main()
