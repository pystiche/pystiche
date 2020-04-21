import os
import subprocess
from os import path

import pdf2image


def run_with_pdflatex(*args, cwd=None):
    subprocess.check_call(("pdflatex", *args), cwd=cwd)


def pdflatex_is_available(cwd=None):
    try:
        run_with_pdflatex("--help", cwd=cwd)
        return True
    except subprocess.CalledProcessError:
        return False


def build_block_diagrams(root, file, dpi=600):
    name, ext = path.splitext(file)

    try:
        run_with_pdflatex("-interaction=nonstopmode", file, cwd=root)
    except subprocess.CalledProcessError:
        # TODO
        raise RuntimeError("pdflatex failed")

    images = pdf2image.convert_from_path(
        path.join(root, f"{name}.pdf"), dpi=dpi, transparent=True
    )
    if len(images) > 1:
        raise RuntimeError("PDF file comprises more than one page.")

    images[0].save(path.join(root, f"{name}.png"))


def main(root):
    if not pdflatex_is_available(cwd=root):
        # TODO
        raise RuntimeError

    for file in os.listdir(root):
        if not file.endswith(".tex"):
            continue

        build_block_diagrams(root, file)


if __name__ == "__main__":
    root = path.dirname(__file__)
    main(root)
