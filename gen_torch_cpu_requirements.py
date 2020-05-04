import argparse
import platform
import re
import sys
from collections import namedtuple
from urllib.parse import urljoin
from urllib.request import urlopen

WHL_PROPS = ("distribution", "version", "language", "abi", "platform")

Whl = namedtuple("whl", (*WHL_PROPS, "url"))


def extract_whl_urls(
    base="https://download.pytorch.org/whl/", html="torch_stable.html"
):
    content = urlopen(urljoin(base, html)).read().decode("utf-8")
    pattern = re.compile('^<a href="[^"]*">(?P<whl>[^<]*)</a><br>$')
    whls = []
    for html_group in content.split("\n"):
        match = pattern.match(html_group)
        if match is not None:
            whls.append(match.group("whl"))
    return whls


def get_torch_cpu_pattern():
    distribution = r"(?P<distribution>torch(vision)?)"
    version = r"(?P<version>\d+[.]\d+[.]\d+([.]post\d+)?)"
    language = r"(?P<language>\w+)"
    abi = r"(?P<abi>\w+)"
    platform = r"(?P<platform>\w+)"
    pattern = re.compile(
        fr"cpu/{distribution}-{version}(%2Bcpu)?-{language}-{abi}-{platform}[.]whl"
    )

    if set(pattern.groupindex.keys()) == set(WHL_PROPS):
        return pattern

    # TODO: include message
    raise RuntimeError


def extract_whls(urls, base="https://download.pytorch.org/whl/"):
    pattern = get_torch_cpu_pattern()

    whls = []
    for url in urls:
        match = pattern.match(url)
        if match is not None:
            kwargs = {prop: match.group(prop) for prop in WHL_PROPS}
            kwargs["url"] = urljoin(base, url)
            whls.append(Whl(**kwargs))

    return whls


def select_link(whls, distribution, language, abi, platform):
    def select(whls, attr, val):
        selected_whls = [whl for whl in whls if getattr(whl, attr) == val]
        if selected_whls:
            return selected_whls

        # TODO: include message
        # valid_vals = set([getattr(whl, attr) for whl in whls])
        raise RuntimeError

    whls = select(whls, "distribution", distribution)
    whls = select(whls, "language", language)
    if abi is not None:
        whls = select(whls, "abi", abi)
    whls = select(whls, "platform", platform)

    return sorted(whls, key=lambda whl: whl.version)[-1].url


def main(args):
    urls = extract_whl_urls()
    whls = extract_whls(urls)

    with open(args.file, "w") as txtfh:
        for distribution in ("torch", "torchvision"):
            link = select_link(
                whls, distribution, args.language, args.abi, args.platform
            )
            txtfh.write(f"{link}\n")


def get_language():
    major, minor, *_ = sys.version_info
    return f"cp{major}{minor}"


def get_platform():
    system = platform.system()
    if system == "Linux":
        return "linux_x86_64"
    elif system == "Windows":
        return "win_amd64"
    elif system == "Darwin":
        major, minor, patch = platform.mac_ver()[0].split(".")
        return f"macosx_{major}_{minor}_x86_64"
    else:
        msg = (
            f"System '{system}' is not recognized. Try setting it manually with "
            "--platform"
        )
        raise RuntimeError(msg)


def parse_input():
    parser = argparse.ArgumentParser(
        description="Generation of a pip requirements file for the latest torch CPU distributions."
    )
    parser.add_argument(
        "--file",
        "-f",
        metavar="PATH",
        type=str,
        default="torch_cpu_requirements.txt",
        help="Path to the pip requirements file to be generated. Defaults to 'torch_cpu_requirements.txt'.",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Language implementation and version tag tag e.g. 'py3', 'cp36'.",
    )
    parser.add_argument(
        "--abi",
        "-a",
        type=str,
        default=None,
        # TODO: describe what is done if not given
        help="Application binary interface (abi) tag e.g. 'cp33m', 'abi3', 'none'.",
    )
    parser.add_argument(
        "--platform",
        "-p",
        type=str,
        default=None,
        help="Platform tag e.g. 'linux_x86_64', 'any'. Defaults to the platform that is used to run this.",
    )
    args = parser.parse_args()

    if args.language is None:
        args.language = get_language()
    if args.platform is None:
        args.platform = get_platform()
    return args


if __name__ == "__main__":
    args = parse_input()
    main(args)
