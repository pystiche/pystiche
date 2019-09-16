from typing import Optional, Tuple, List
import numpy as np
import torch
from torch.nn.functional import affine_grid, grid_sample
from pystiche.typing import Numeric
from pystiche.misc import verify_str_arg
from pystiche.image.utils import extract_image_size

__all__ = [
    "create_affine_transformation_matrix",
    "transform_motif_affinely",
    "shear_motif",
    "rotate_motif",
    "scale_motif",
    "translate_motif",
    "resize_canvas",
]


def _create_motif_shearing_matrix(
    angle: Numeric, clockwise: bool = False
) -> torch.Tensor:
    angle = np.deg2rad(angle)
    if clockwise:
        angle *= -1.0
    shearing_matrix = (
        (1.0, -np.sin(angle), 0.0),
        (0.0, np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    return torch.tensor(shearing_matrix)


def _create_motif_rotation_matrix(
    angle: Numeric, clockwise: bool = False
) -> torch.Tensor:
    angle = np.deg2rad(angle)
    if clockwise:
        angle *= -1.0
    rotation_matrix = (
        (np.cos(angle), -np.sin(angle), 0.0),
        (np.sin(angle), np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    return torch.tensor(rotation_matrix)


def _create_motif_scaling_matrix(factors: Tuple[Numeric, Numeric]) -> torch.Tensor:
    factor_vert, factor_horz = factors
    scaling_matrix = ((factor_horz, 0.0, 0.0), (0.0, factor_vert, 0.0), (0.0, 0.0, 1.0))
    return torch.tensor(scaling_matrix)


def _create_motif_translation_matrix(
    translation: Tuple[Numeric, Numeric], inverse: bool = False
) -> torch.Tensor:
    if inverse:
        translation = [-val for val in translation]
    translation_vert, translation_horz = translation
    translation_matrix = (
        (1.0, 0.0, translation_horz),
        (0.0, 1.0, translation_vert),
        (0.0, 0.0, 1.0),
    )
    return torch.tensor(translation_matrix)


def _calculate_image_center(image_size: Tuple[int, int]) -> Tuple[float, float]:
    return tuple([edge_size / 2.0 for edge_size in image_size])


def _transform_around_point(
    point: Tuple[float, float], transformation_matrix: torch.Tensor
) -> torch.Tensor:
    return torch.chain_matmul(
        _create_motif_translation_matrix(point, inverse=False),
        transformation_matrix,
        _create_motif_translation_matrix(point, inverse=True),
    )


def create_affine_transformation_matrix(
    image_size: Tuple[int, int],
    shearing_angle: Optional[Numeric] = None,
    clockwise_shearing: bool = False,
    shearing_center: Optional[Tuple[Numeric, Numeric]] = None,
    rotation_angle: Optional[Numeric] = None,
    clockwise_rotation: bool = False,
    rotation_center: Optional[Tuple[Numeric, Numeric]] = None,
    scaling_factors: Optional[Tuple[Numeric, Numeric]] = None,
    scaling_center: Optional[Tuple[Numeric, Numeric]] = None,
    translation: Optional[Tuple[Numeric, Numeric]] = None,
    inverse_translation: bool = False,
) -> torch.Tensor:
    if not any((shearing_angle, rotation_angle, scaling_factors, translation)):
        raise RuntimeError

    image_center = _calculate_image_center(image_size)
    if shearing_center is None:
        shearing_center = image_center
    if rotation_center is None:
        rotation_center = image_center
    if scaling_center is None:
        scaling_center = image_center

    transformation_matrices = []
    if shearing_angle is not None:
        transform_matrix = _create_motif_shearing_matrix(
            shearing_angle, clockwise=clockwise_shearing
        )
        transform_matrix = _transform_around_point(shearing_center, transform_matrix)
        transformation_matrices.append(transform_matrix)
    if rotation_angle is not None:
        transform_matrix = _create_motif_rotation_matrix(
            rotation_angle, clockwise=clockwise_rotation
        )
        transform_matrix = _transform_around_point(rotation_center, transform_matrix)
        transformation_matrices.append(transform_matrix)
    if scaling_factors is not None:
        transform_matrix = _create_motif_scaling_matrix(scaling_factors)
        transform_matrix = _transform_around_point(scaling_center, transform_matrix)
        transformation_matrices.append(transform_matrix)
    if translation is not None:
        transform_matrix = _create_motif_translation_matrix(
            translation, inverse=inverse_translation
        )
        transformation_matrices.append(transform_matrix)

    return torch.chain_matmul(*reversed(transformation_matrices))


def transform_motif_affinely(
    x: torch.Tensor,
    transformation_matrix: torch.Tensor,
    output_image_size: Optional[Tuple[int, int]] = None,
    interpolation_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    batch_size, num_channels = x.size()[:2]
    input_image_size = extract_image_size(x)
    transformation_matrix = _transform_coordinates(
        transformation_matrix, input_image_size
    )

    if output_image_size is None:
        output_image_size = input_image_size
    output_size = [batch_size, num_channels] + list(output_image_size)

    grid = _calculate_affine_grid(transformation_matrix, output_size)
    return grid_sample(
        x, grid.to(x.device), mode=interpolation_mode, padding_mode=padding_mode
    )


def _transform_coordinates(
    transformation_matrix: torch.Tensor, image_size: Tuple[int, int]
) -> torch.Tensor:
    vert_center, horz_center = _calculate_image_center(image_size)
    coordinate_transformation_matrix = (
        (horz_center, 0.0, horz_center),
        (0.0, -vert_center, vert_center),
        (0.0, 0.0, 1.0),
    )
    coordinate_transformation_matrix = torch.tensor(coordinate_transformation_matrix)
    return torch.chain_matmul(
        torch.inverse(coordinate_transformation_matrix),
        transformation_matrix,
        coordinate_transformation_matrix,
    )


def _calculate_affine_grid(
    transformation_matrix: torch.Tensor, size: List[int]
) -> torch.Tensor:
    inv_transformation_matrix = torch.inverse(transformation_matrix)
    theta = inv_transformation_matrix[:-1, :].unsqueeze(0)
    return affine_grid(theta, size)


def shear_motif(
    x: torch.Tensor,
    angle: Numeric,
    clockwise: bool = False,
    center: Optional[Tuple[Numeric, Numeric]] = None,
    **kwargs
) -> torch.Tensor:
    image_size = extract_image_size(x)
    transformation_matrix = create_affine_transformation_matrix(
        image_size,
        shearing_angle=angle,
        clockwise_shearing=clockwise,
        shearing_center=center,
    )
    return transform_motif_affinely(x, transformation_matrix, **kwargs)


def rotate_motif(
    x: torch.Tensor,
    angle: Numeric,
    clockwise: bool = False,
    center: Optional[Tuple[Numeric, Numeric]] = None,
    **kwargs
) -> torch.Tensor:
    image_size = extract_image_size(x)
    transformation_matrix = create_affine_transformation_matrix(
        image_size,
        rotation_angle=angle,
        clockwise_rotation=clockwise,
        rotation_center=center,
    )
    return transform_motif_affinely(x, transformation_matrix, **kwargs)


def scale_motif(
    x: torch.Tensor,
    factors: Tuple[Numeric, Numeric],
    center: Optional[Tuple[Numeric, Numeric]] = None,
    **kwargs
) -> torch.Tensor:
    image_size = extract_image_size(x)
    transformation_matrix = create_affine_transformation_matrix(
        image_size, scaling_factors=factors, scaling_center=center
    )
    return transform_motif_affinely(x, transformation_matrix, **kwargs)


def translate_motif(
    x: torch.Tensor,
    translation: Tuple[Numeric, Numeric],
    inverse: bool = False,
    **kwargs
) -> torch.Tensor:
    image_size = extract_image_size(x)
    transformation_matrix = create_affine_transformation_matrix(
        image_size, translation=translation, inverse_translation=inverse
    )
    return transform_motif_affinely(x, transformation_matrix, **kwargs)


def _calculate_full_bounding_box_size(vertices: torch.Tensor) -> Tuple[int, int]:
    # TODO: rename x and y
    x, y = torch.max(torch.abs(vertices), 1).values.tolist()
    return tuple([int(np.ceil(2.0 * val)) for val in (y, x)])


def _calculate_valid_bounding_box_size(vertices):
    raise RuntimeError


def resize_canvas(
    transformation_matrix: torch.Tensor,
    image_size: Tuple[int, int],
    method: str = "full",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    verify_str_arg(method, "method", ("same", "full", "valid"))

    if method == "same":
        return transformation_matrix, image_size

    def center_motif(transformation_matrix, image_size):
        image_center = _calculate_image_center(image_size)
        image_center = torch.tensor((*image_center[::-1], 1.0)).unsqueeze(1)
        motif_center = torch.mm(transformation_matrix, image_center)
        motif_center = motif_center[:-1, 0].tolist()[::-1]

        translation_matrix = _create_motif_translation_matrix(
            motif_center, inverse=True
        )
        return torch.mm(translation_matrix, transformation_matrix)

    def calculate_motif_vertices(transformation_matrix, image_size):
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
        return torch.mm(transformation_matrix, image_vertices)[:-1, :]

    def scale_and_off_center_motif(
        transformation_matrix, image_size, bounding_box_size
    ):
        height, width = image_size
        image_center = _calculate_image_center(image_size)

        bounding_box_height, bounding_box_width = bounding_box_size
        scaling_factors = (height / bounding_box_height, width / bounding_box_width)
        scaling_matrix = _create_motif_scaling_matrix(scaling_factors)
        scaling_matrix = _transform_around_point(image_center, scaling_matrix)

        translation_matrix = _create_motif_translation_matrix(image_center)

        return torch.chain_matmul(
            scaling_matrix, translation_matrix, transformation_matrix
        )

    transformation_matrix = center_motif(transformation_matrix, image_size)
    motif_vertices = calculate_motif_vertices(transformation_matrix, image_size)

    if method == "full":
        bounding_box_size = _calculate_full_bounding_box_size(motif_vertices)
    else:  # method == "valid"
        bounding_box_size = _calculate_valid_bounding_box_size(motif_vertices)

    transformation_matrix = scale_and_off_center_motif(
        transformation_matrix, image_size, bounding_box_size
    )

    return transformation_matrix, bounding_box_size
