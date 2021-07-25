from typing import Any, Dict, Iterator, Optional

import torch
from torchvision.transforms.functional import resize

from pystiche import ComplexObject
from pystiche._compat import InterpolationMode
from pystiche.image import edge_to_image_size, extract_aspect_ratio
from pystiche.misc import verify_str_arg

__all__ = ["PyramidLevel"]


class PyramidLevel(ComplexObject):
    r"""Level with an :class:`pystiche.pyramid.ImagePyramid`. If iterated on, yields
    the step beginning at 1 and ending in ``num_steps``.

    Args:
        edge_size: Edge size.
        num_steps: Number of steps.
        edge: Corresponding edge to the edge size. Can be ``"short"`` or
            ``"long"``.
    """

    def __init__(self, edge_size: int, num_steps: int, edge: str) -> None:
        self.edge_size = edge_size
        self.num_steps = num_steps
        self.edge = verify_str_arg(edge, "edge", ("short", "long"))

    def _resize(
        self,
        image: torch.Tensor,
        aspect_ratio: Optional[float],
        interpolation_mode: str,
    ) -> torch.Tensor:
        if aspect_ratio is None:
            aspect_ratio = extract_aspect_ratio(image)
        image_size = edge_to_image_size(self.edge_size, aspect_ratio, edge=self.edge)

        with torch.no_grad():
            image = resize(
                image,
                list(image_size),
                interpolation=InterpolationMode(interpolation_mode),
            )
        return image.detach()

    def resize_image(
        self,
        image: torch.Tensor,
        aspect_ratio: Optional[float] = None,
        interpolation_mode: str = "bilinear",
    ) -> torch.Tensor:
        r"""Resize an image to the ``edge_size`` on the corresponding ``edge`` of the
        :class:`PyramidLevel`.

        Args:
            image: Image to be resized.
            aspect_ratio: Optional aspect ratio of the output. If ``None``, the aspect
                ratio of ``image`` is used. Defaults to ``None``.
            interpolation_mode: Interpolation mode used to resize ``image``. Defaults
                to ``"bilinear"``.

        .. warning::

            The resizing is performed without gradient calculation. Do **not** use this
            if the image needs a gradient.
        """
        return self._resize(image, aspect_ratio, interpolation_mode)

    def resize_guide(
        self,
        guide: torch.Tensor,
        aspect_ratio: Optional[float] = None,
        interpolation_mode: str = "nearest",
    ) -> torch.Tensor:
        r"""Resize a guide to the ``edge_size`` on the corresponding ``edge`` of the
        :class:`PyramidLevel`.

        Args:
            guide: Guide to be resized.
            aspect_ratio: Optional aspect ratio of the output. If ``None``, the aspect
                ratio of ``guide`` is used. Defaults to ``None``.
            interpolation_mode: Interpolation mode used to resize ``image``. Defaults
                to ``"nearest"``.
        """
        return self._resize(guide, aspect_ratio, interpolation_mode)

    def __iter__(self) -> Iterator[int]:
        yield from range(1, self.num_steps + 1)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["edge_size"] = self.edge_size
        dct["num_steps"] = self.num_steps
        dct["edge"] = self.edge
        return dct
