from typing import Any, Union, Optional, Sequence
from pystiche.pyramid import OctaveImagePyramid

__all__ = ["li_wand_2016_image_pyramid"]


def li_wand_2016_image_pyramid(
    impl_params: bool = True,
    max_edge_size: int = 384,
    num_steps: Optional[Union[int, Sequence[int]]] = None,
    num_levels: Optional[int] = None,
    min_edge_size: int = 64,
    edge: Union[str, Sequence[str]] = "long",
    **octave_image_pyramid_kwargs: Any,
):
    if num_steps is None:
        num_steps = 100 if impl_params else 200

    if num_levels is None:
        num_levels = 3 if impl_params else None

    return OctaveImagePyramid(
        max_edge_size,
        num_steps,
        num_levels=num_levels,
        min_edge_size=min_edge_size,
        edge=edge,
        **octave_image_pyramid_kwargs,
    )
