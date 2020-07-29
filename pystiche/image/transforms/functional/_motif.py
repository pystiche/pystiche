from typing import NoReturn, Optional, Tuple, Union, cast

import numpy as np

import torch

from pystiche.image.utils import (
    extract_batch_size,
    extract_image_size,
    extract_num_channels,
    force_batched_image,
)
from pystiche.misc import to_2d_arg, verify_str_arg

from ._align_corners import affine_grid, grid_sample

__all__ = [
    "transform_motif_affinely",
    "shear_motif",
    "rotate_motif",
    "scale_motif",
    "translate_motif",
]


def _create_motif_shearing_matrix(
    angle: float, clockwise: bool = False
) -> torch.Tensor:
    angle = np.deg2rad(angle)
    if clockwise:
        angle *= -1.0
    shearing_matrix = (
        (1.0, -np.sin(angle), 0.0),
        (0.0, np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    return torch.tensor(shearing_matrix, dtype=torch.float32)


def _create_motif_rotation_matrix(
    angle: float, clockwise: bool = False
) -> torch.Tensor:
    angle = np.deg2rad(angle)
    if clockwise:
        angle *= -1.0
    rotation_matrix = (
        (np.cos(angle), -np.sin(angle), 0.0),
        (np.sin(angle), np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    return torch.tensor(rotation_matrix, dtype=torch.float32)


def _create_motif_scaling_matrix(
    factor: Union[float, Tuple[float, float]]
) -> torch.Tensor:
    factor_vert, factor_horz = to_2d_arg(factor)
    scaling_matrix = ((factor_horz, 0.0, 0.0), (0.0, factor_vert, 0.0), (0.0, 0.0, 1.0))
    return torch.tensor(scaling_matrix, dtype=torch.float32)


def _create_motif_translation_matrix(
    translation: Tuple[float, float], inverse: bool = False
) -> torch.Tensor:
    translation_vert, translation_horz = translation
    if inverse:
        translation_vert = -translation_vert
        translation_horz = -translation_horz
    translation_matrix = (
        (1.0, 0.0, translation_horz),
        (0.0, 1.0, translation_vert),
        (0.0, 0.0, 1.0),
    )
    return torch.tensor(translation_matrix, dtype=torch.float32)


def _calculate_image_center(image_size: Tuple[int, int]) -> Tuple[float, float]:
    height, width = image_size
    vert_center = height / 2.0
    horz_center = width / 2.0
    return vert_center, horz_center


def _transform_around_point(
    point: Tuple[float, float], transform_matrix: torch.Tensor
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        torch.chain_matmul(
            _create_motif_translation_matrix(point, inverse=False),
            transform_matrix,
            _create_motif_translation_matrix(point, inverse=True),
        ),
    )


def _transform_coordinates(
    transform_matrix: torch.Tensor, image_size: Tuple[int, int]
) -> torch.Tensor:
    vert_center, horz_center = _calculate_image_center(image_size)
    coordinate_transform_matrix = (
        (horz_center, 0.0, horz_center),
        (0.0, -vert_center, vert_center),
        (0.0, 0.0, 1.0),
    )
    coordinate_transform_matrix = torch.tensor(coordinate_transform_matrix)
    return cast(
        torch.Tensor,
        torch.chain_matmul(
            torch.inverse(coordinate_transform_matrix),
            transform_matrix,
            coordinate_transform_matrix,
        ),
    )


def _create_affine_transform_matrix(
    image_size: Tuple[int, int],
    shearing_angle: Optional[float] = None,
    clockwise_shearing: bool = False,
    shearing_center: Optional[Tuple[float, float]] = None,
    rotation_angle: Optional[float] = None,
    clockwise_rotation: bool = False,
    rotation_center: Optional[Tuple[float, float]] = None,
    scaling_factor: Optional[Union[float, Tuple[float, float]]] = None,
    scaling_center: Optional[Tuple[float, float]] = None,
    translation: Optional[Tuple[float, float]] = None,
    inverse_translation: bool = False,
) -> torch.Tensor:
    if not any((shearing_angle, rotation_angle, scaling_factor, translation)):
        return torch.eye(3)

    image_center = _calculate_image_center(image_size)
    if shearing_center is None:
        shearing_center = image_center
    if rotation_center is None:
        rotation_center = image_center
    if scaling_center is None:
        scaling_center = image_center

    transform_matrices = []
    if shearing_angle is not None:
        transform_matrix = _create_motif_shearing_matrix(
            shearing_angle, clockwise=clockwise_shearing
        )
        transform_matrix = _transform_around_point(shearing_center, transform_matrix)
        transform_matrices.append(transform_matrix)
    if rotation_angle is not None:
        transform_matrix = _create_motif_rotation_matrix(
            rotation_angle, clockwise=clockwise_rotation
        )
        transform_matrix = _transform_around_point(rotation_center, transform_matrix)
        transform_matrices.append(transform_matrix)
    if scaling_factor is not None:
        transform_matrix = _create_motif_scaling_matrix(scaling_factor)
        transform_matrix = _transform_around_point(scaling_center, transform_matrix)
        transform_matrices.append(transform_matrix)
    if translation is not None:
        transform_matrix = _create_motif_translation_matrix(
            translation, inverse=inverse_translation
        )
        transform_matrices.append(transform_matrix)

    return cast(torch.Tensor, torch.chain_matmul(*reversed(transform_matrices)))


def _calculate_full_bounding_box_size(vertices: torch.Tensor) -> Tuple[int, int]:
    # When calling torch.max with a dimension as second argument, a namedtuple with a
    # values field is returned.
    # https://pytorch.org/docs/stable/torch.html#torch.max
    # This is not reflected in the type hints.
    coords = torch.max(torch.abs(vertices), 1).values
    return cast(
        Tuple[int, int],
        tuple(reversed(torch.ceil(2.0 * coords).to(torch.int).tolist())),
    )


def _calculate_valid_bounding_box_size(vertices: torch.Tensor) -> NoReturn:
    msg = "The valid canvas option is not yet implemented."
    raise RuntimeError(msg)


def _resize_canvas(
    transform_matrix: torch.Tensor, image_size: Tuple[int, int], method: str = "same",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    verify_str_arg(method, "method", ("same", "full", "valid"))

    if method == "same":
        return transform_matrix, image_size

    def center_motif(
        transform_matrix: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        image_center = _calculate_image_center(image_size)
        image_center = torch.tensor((*image_center[::-1], 1.0)).unsqueeze(1)
        motif_center = torch.mm(transform_matrix, image_center)
        motif_center = cast(Tuple[float, float], motif_center[:-1, 0].tolist()[::-1])

        translation_matrix = _create_motif_translation_matrix(
            motif_center, inverse=True
        )
        return torch.mm(translation_matrix, transform_matrix)

    def calculate_motif_vertices(
        transform_matrix: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        height, width = image_size
        # TODO: do this without transpose
        image_vertices = torch.tensor(
            (
                (0.0, 0.0, 1.0),
                (width, 0.0, 1.0),
                (0.0, height, 1.0),
                (width, height, 1.0),
            )
        ).t()
        return torch.mm(transform_matrix, image_vertices)[:-1, :]

    def scale_and_off_center_motif(
        transform_matrix: torch.Tensor,
        image_size: Tuple[int, int],
        bounding_box_size: Tuple[int, int],
    ) -> torch.Tensor:
        height, width = image_size
        image_center = _calculate_image_center(image_size)

        bounding_box_height, bounding_box_width = bounding_box_size
        scaling_factors = (height / bounding_box_height, width / bounding_box_width)
        scaling_matrix = _create_motif_scaling_matrix(scaling_factors)
        scaling_matrix = _transform_around_point(image_center, scaling_matrix)

        translation_matrix = _create_motif_translation_matrix(image_center)

        return cast(
            torch.Tensor,
            torch.chain_matmul(scaling_matrix, translation_matrix, transform_matrix),
        )

    transform_matrix = center_motif(transform_matrix, image_size)
    motif_vertices = calculate_motif_vertices(transform_matrix, image_size)

    if method == "full":
        bounding_box_size = _calculate_full_bounding_box_size(motif_vertices)
    else:  # method == "valid"
        bounding_box_size = _calculate_valid_bounding_box_size(motif_vertices)

    transform_matrix = scale_and_off_center_motif(
        transform_matrix, image_size, bounding_box_size
    )

    return transform_matrix, bounding_box_size


def _calculate_affine_grid(
    image: torch.Tensor, transform_matrix: torch.Tensor, image_size: Tuple[int, int]
) -> torch.Tensor:
    batch_size = extract_batch_size(image)
    num_channels = extract_num_channels(image)
    height, width = image_size
    size = (batch_size, num_channels, height, width)

    inv_transform_matrix = torch.inverse(transform_matrix)
    theta = inv_transform_matrix[:-1, :].unsqueeze(0)

    return affine_grid(theta, size)


@force_batched_image
def transform_motif_affinely(
    image: torch.Tensor,
    shearing_angle: Optional[float] = None,
    clockwise_shearing: bool = False,
    shearing_center: Optional[Tuple[float, float]] = None,
    rotation_angle: Optional[float] = None,
    clockwise_rotation: bool = False,
    rotation_center: Optional[Tuple[float, float]] = None,
    scaling_factor: Optional[Union[float, Tuple[float, float]]] = None,
    scaling_center: Optional[Tuple[float, float]] = None,
    translation: Optional[Tuple[float, float]] = None,
    inverse_translation: bool = False,
    canvas: str = "same",
    interpolation_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    device = image.device
    image_size = extract_image_size(image)

    transform_matrix = _create_affine_transform_matrix(
        image_size,
        shearing_angle=shearing_angle,
        clockwise_shearing=clockwise_shearing,
        shearing_center=shearing_center,
        rotation_angle=rotation_angle,
        clockwise_rotation=clockwise_rotation,
        rotation_center=rotation_center,
        scaling_factor=scaling_factor,
        scaling_center=scaling_center,
        translation=translation,
        inverse_translation=inverse_translation,
    )
    transform_matrix, resized_image_size = _resize_canvas(
        transform_matrix, image_size, method=canvas
    )
    transform_matrix = _transform_coordinates(transform_matrix, image_size)
    transform_matrix = transform_matrix.to(device)

    grid = _calculate_affine_grid(image, transform_matrix, resized_image_size)
    grid = grid.to(device)

    return grid_sample(image, grid, mode=interpolation_mode, padding_mode=padding_mode,)


def shear_motif(
    image: torch.Tensor,
    angle: float,
    clockwise: bool = False,
    center: Optional[Tuple[float, float]] = None,
    canvas: str = "same",
    interpolation_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        transform_motif_affinely(
            image,
            shearing_angle=angle,
            clockwise_shearing=clockwise,
            shearing_center=center,
            canvas=canvas,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
        ),
    )


def rotate_motif(
    image: torch.Tensor,
    angle: float,
    clockwise: bool = False,
    center: Optional[Tuple[float, float]] = None,
    canvas: str = "same",
    interpolation_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        transform_motif_affinely(
            image,
            rotation_angle=angle,
            clockwise_rotation=clockwise,
            rotation_center=center,
            canvas=canvas,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
        ),
    )


def scale_motif(
    image: torch.Tensor,
    factor: Union[float, Tuple[float, float]],
    center: Optional[Tuple[float, float]] = None,
    canvas: str = "same",
    interpolation_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        transform_motif_affinely(
            image,
            scaling_factor=factor,
            scaling_center=center,
            canvas=canvas,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
        ),
    )


def translate_motif(
    image: torch.Tensor,
    translation: Tuple[float, float],
    inverse: bool = False,
    canvas: str = "same",
    interpolation_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        transform_motif_affinely(
            image,
            translation=translation,
            inverse_translation=inverse,
            canvas=canvas,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
        ),
    )
