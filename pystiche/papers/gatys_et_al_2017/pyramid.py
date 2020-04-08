from typing import Sequence, Union

from pystiche.pyramid import ImagePyramid

__all__ = ["gatys_et_al_2017_image_pyramid"]


def gatys_et_al_2017_image_pyramid(
    edge_sizes: Sequence[int] = (500, 800),
    num_steps: Union[int, Sequence[int]] = (500, 200),
    **image_pyramid_kwargs,
):
    return ImagePyramid(edge_sizes, num_steps, **image_pyramid_kwargs)
