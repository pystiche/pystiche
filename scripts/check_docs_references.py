import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

WARNING_PATTERN = r"WARNING: (?P<warning>.*?)$"
FILE_PATTERN = re.compile(r"(?P<location>.*?): " + WARNING_PATTERN)
DOCSTRING_PATTERN = re.compile(r".*?(?P<location>docstring of .*?): " + WARNING_PATTERN)

IGNORED_WARNINGS = {
    "py:class reference target not found: torchvision.models.VGG",
    "py:class reference target not found: torchvision.models.AlexNet",
    "py:mod reference target not found: torch",
    "py:class reference target not found: torch.device",
    "py:class reference target not found: torch.nn.modules.module._IncompatibleKeys",
    "py:class reference target not found: torch.optim.lr_scheduler._LRScheduler",
    "py:class reference target not found: Module",
    # every warning below this comment should be only temporary ignored and can be
    # resolved quickly
    "py:attr reference target not found: transform",
    "py:exc reference target not found: framework`",
    "py:class reference target not found: OperatorContainer",
    "py:meth reference target not found: pystiche.ops.Comparison.set_input_guide",
    "py:meth reference target not found: pystiche.ops.Comparison.set_target_image",
    "py:class reference target not found: pystiche.loss._comparison.ScaleAndRotate",
    "py:class reference target not found: pystiche.ops.TotalVariationOperator",
    "py:exc reference target not found: getter.",
    "py:class reference target not found: pystiche.loss.GuidedPerceptualLoss",
    "py:meth reference target not found: pystiche.loss.PerceptualLoss.set_content_image",
    "py:class reference target not found: pystiche.ops.Operator",
    "py:class reference target not found: pystiche.ops.RegularizationOperator",
    "py:class reference target not found: pystiche.ops.ComparisonOperator",
    "py:class reference target not found: pystiche.ops.op.PixelOperator",
    "py:class reference target not found: pystiche.ops.EncodingOperator",
    "py:class reference target not found: pystiche.ops.PixelRegularizationOperator",
    "py:class reference target not found: pystiche.ops.TotalVariationOperator",
    "py:class reference target not found: pystiche.ops.EncodingRegularizationOperator",
    "py:class reference target not found: pystiche.ops.PixelComparisonOperator",
    "py:class reference target not found: pystiche.ops.EncodingComparisonOperator",
    "py:class reference target not found: pystiche.ops.FeatureReconstructionOperator",
    "py:class reference target not found: pystiche.ops.GramOperator",
    "py:class reference target not found: pystiche.ops.MRFOperator",
    "py:class reference target not found: pystiche.ops.MultiLayerEncodingOperator",
    "py:class reference target not found: pystiche.ops.PixelOperator",
}


def main(source_dir):
    warnings = capture_warning_output(source_dir)
    valid_warnings, unused_ignores = filter_warnings(warnings)
    if not (valid_warnings or unused_ignores):
        sys.exit(0)

    if valid_warnings:
        print("Building the documentation emitted these warnings:", file=sys.stderr)
        for location, warning in valid_warnings:
            print(f"{location}: {warning}", file=sys.stderr)

    if unused_ignores:
        print("These warnings were ignored, but not detected:,", file=sys.stderr)
        for warning in sorted(unused_ignores):
            print(warning, file=sys.stderr)

    sys.exit(1)


def capture_warning_output(source_dir):
    source_dir = pathlib.Path(source_dir).expanduser().resolve()
    build_dir = tempfile.mkdtemp()

    env = os.environ.copy()
    env.setdefault("PYSTICHE_PLOT_GALLERY", str(False))

    output = subprocess.check_output(
        ("sphinx-build", "-nqb", "html", source_dir, build_dir,),
        env=env,
        stderr=subprocess.STDOUT,
    )

    shutil.rmtree(build_dir)

    return output.decode().strip().splitlines()


def filter_warnings(warnings):
    valid_warnings = []
    used_ignores = set()

    for line in warnings:
        pattern = DOCSTRING_PATTERN if "docstring of" in line else FILE_PATTERN

        match = pattern.match(line)
        if match is None:
            raise RuntimeError(
                f"The following line does not match the pattern to identify the "
                f"location and message of a warning:\n\n"
                f"{line}"
            )

        location = match.group("location")
        warning = match.group("warning")

        if warning in IGNORED_WARNINGS:
            used_ignores.add(warning)
            continue

        valid_warnings.append((location, warning))

    return valid_warnings, IGNORED_WARNINGS - used_ignores


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(
            "Please supply the documentation source directory as positional argument"
        )
    main(sys.argv[1])
