import itertools
import os
from functools import reduce
from os import path
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch

from .io import read_image, write_image
from .utils import extract_image_size, extract_num_channels, force_single_image

__all__ = [
    "verify_guides",
    "read_guides",
    "write_guides",
    "guides_to_segmentation",
    "segmentation_to_guides",
]

# Color are taken from
# https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
# fmt: off
COLOR_ORDER = (
    (166, 206, 227),
    ( 31, 120, 180),  # noqa: E201
    (178, 223, 138),
    ( 51, 160,  44),  # noqa: E201, E241
    (251, 154, 153),
    (227,  26,  28),  # noqa: E241
    (253, 191, 111),
    (255, 127,   0),  # noqa: E241
    (202, 178, 214),
    (106,  61, 154),  # noqa: E241
    (255, 255, 153),
    (177,  89,  40),  # noqa: E241
)
# fmt: on

RGBTriplet = Tuple[int, int, int]


def verify_guides(
    guides: Dict[str, torch.Tensor],
    verify_coverage: bool = True,
    verify_overlap: bool = True,
) -> None:
    """Verify if guides cover the whole canvas and if they do not overlap with each
    other.

    Args:
        guides: Guides to be verified.
        verify_coverage: If ``True`` verifies that the guides cover the whole canvas.
            Defaults to ``True``.
        verify_overlap: If ``True`` verifies that the guides do not overlap with each
            other. Defaults to ``True``.

    Raises:
        RuntimeError: If at least on check is not successful.
    """
    if not (verify_coverage or verify_overlap):
        return

    masks = {region: guide.bool() for region, guide in guides.items()}

    if verify_coverage:
        coverage = reduce(lambda mask1, mask2: mask1 | mask2, masks.values())
        if not torch.all(coverage):
            numel = coverage.numel()
            abs_miss = numel - torch.sum(coverage)
            rel_miss = abs_miss.float() / numel
            msg = f"{abs_miss} pixels ({rel_miss:.1%}) are not covered by the guides."
            raise RuntimeError(msg)

    if verify_overlap:
        overlaps = []
        for (region1, mask1), (region2, mask2) in itertools.combinations(
            masks.items(), 2
        ):
            overlap = mask1 & mask2
            if torch.any(overlap):
                overlaps.append(f"{region1}, {region2}: {torch.sum(overlap)} pixels")

        if overlaps:
            msg = "\n".join(
                (
                    "The guides in the following regions overlap with each other:",
                    "",
                    *overlaps,
                )
            )
            raise RuntimeError(msg)


def read_guides(
    dir: str,
    device: Union[torch.device, str] = "cpu",
    make_batched: bool = True,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    interpolation_mode: str = "nearest",
) -> Dict[str, torch.Tensor]:
    """Read all guides from a directory using :func:`~pystiche.image.read_image` and
    return them as dictionary. The filename without extensions is used as region key.

    Args:
        dir: Path to root directory of the guide files.
        device: Device that the guides are transferred to. Defaults to CPU.
        make_batched: If ``True``, a fake batch dimension is added to every guide.
        size: Optional size the guides are resized to.
        interpolation_mode: Interpolation mode that is used to perform the optional
            resizing. Valid modes are ``"nearest"``, ``"bilinear"``, and ``"bicubic"``.
            Defaults to ``"nearest"``.
    """

    def read_guide(file: str) -> torch.Tensor:
        return read_image(
            path.join(dir, file),
            device=device,
            make_batched=make_batched,
            size=size,
            interpolation_mode=interpolation_mode,
        )

    return {path.splitext(file)[0]: read_guide(file) for file in os.listdir(dir)}


def write_guides(
    guides: Dict[str, torch.Tensor],
    dir: str,
    ext: str = ".png",
    mode: str = "L",
    **save_kwargs: Any,
) -> None:
    """Write guides to directory using :func:`~pystiche.image.write_image`. The region
    key is used as filename.

    Args:
        guides: Guides to be written.
        dir: Path to root directory of the guide files.
        ext: Extension that is appended to the filename. Defaults to ``".png"``.
        mode: Optional image mode. See the `Pillow documentation
            <https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes>`_
            for details. Defaults to ``"L"``.
        **save_kwargs: Other parameters that are passed to :meth:`PIL.Image.Image.save`
            .
    """
    for region, guide in guides.items():
        file = path.join(dir, region + ext)
        write_image(guide, file, mode=mode, **save_kwargs)


def guides_to_segmentation(
    guides: Dict[str, torch.Tensor], color_map: Optional[Dict[str, RGBTriplet]] = None,
) -> torch.Tensor:
    """Combines multiple guides into one segmentation image.

    Args:
        guides: Guides to be combined.
        color_map: Optional mapping from regions to RGB triplets. If omitted,
            `this color scheme <https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12>`_
            is used for the region keys in alphabetical order.
    """
    if color_map is None:
        color_map = dict(zip(sorted(guides.keys()), COLOR_ORDER))

    image_size = extract_image_size(tuple(guides.values())[0])
    height, width = image_size
    seg_flat = torch.empty(3, height * width, dtype=torch.float32)

    for region, guide in guides.items():
        rgb_triplet = color_map[region]
        color = torch.tensor(rgb_triplet, dtype=torch.float32).div(255.0).view(3, 1)
        seg_flat[:, guide.bool().flatten()] = color

    return seg_flat.view(1, 3, *image_size)


@force_single_image
def segmentation_to_guides(
    seg: torch.Tensor, region_map: Optional[Dict[RGBTriplet, str]] = None
) -> Dict[Union[RGBTriplet, str], torch.Tensor]:
    """Splits a segmentation image in multiple guides.

    Args:
        seg: Segmentation image to be split.
        region_map: Optional mapping from RGB triplets to regions. If omitted, the RGB
            triplets are used as key in the output.
    """
    num_chanels = extract_num_channels(seg)
    if num_chanels != 3:
        msg = f"Expected 3 channels in the segmentation image, but got {num_chanels}."
        raise ValueError(msg)

    image_size = extract_image_size(seg)

    seg_flat = seg.view(3, -1)
    seg_flat = seg_flat.mul(255.0).byte()

    colors = seg_flat.unique(sorted=False, dim=1).split(1, dim=1)
    guides: Dict[Union[RGBTriplet, str], torch.Tensor] = {}
    for color in colors:
        guide_flat = torch.all(seg_flat == color, dim=0)
        guide = guide_flat.float().view(1, 1, *image_size)

        rgb_triplet = cast(RGBTriplet, tuple(color.squeeze().tolist()))
        guides[rgb_triplet] = guide

    if region_map is None:
        return guides

    return {
        region_map.get(rgb_triplet, rgb_triplet): guide
        for rgb_triplet, guide in cast(Dict[RGBTriplet, torch.Tensor], guides).items()
    }
