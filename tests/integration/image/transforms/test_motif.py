import pillow_affine as pa
import pytest
from PIL import Image

import torch

from pystiche.image import transforms

from . import assert_transform_equals_pil


def get_pil_affine_transform(affine_transform, expand=False):
    def transform(image):
        transform_params = affine_transform.extract_transform_params(
            image.size, expand=expand
        )
        return image.transform(*transform_params, resample=Image.BILINEAR)

    return transform


def test_ShearMotif():
    def PILShearMotif(angle, clockwise=False, center=None):
        if center is not None:
            center = tuple(center[::-1])
        affine_transform = pa.Shear(angle, clockwise=clockwise, center=center)
        return get_pil_affine_transform(affine_transform)

    angle = 30
    pystiche_transform = transforms.ShearMotif(angle)
    pil_transform = PILShearMotif(angle)
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform,
    )

    clockwise = True
    pystiche_transform = transforms.ShearMotif(angle, clockwise=clockwise)
    pil_transform = PILShearMotif(angle, clockwise=clockwise)
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform,
    )

    center = (100.0, 50.0)
    pystiche_transform = transforms.ShearMotif(angle, center=center)
    pil_transform = PILShearMotif(angle, center=center)
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform,
    )


def test_RotateMotif():
    def PILRotateMotif(angle, clockwise=False, center=None):
        if center is not None:
            center = tuple(center[::-1])
        affine_transform = pa.Rotate(angle, clockwise=clockwise, center=center)
        return get_pil_affine_transform(affine_transform)

    angle = 30
    pystiche_transform = transforms.RotateMotif(angle)
    pil_transform = PILRotateMotif(angle)
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform,
    )

    clockwise = True
    pystiche_transform = transforms.RotateMotif(angle, clockwise=clockwise)
    pil_transform = PILRotateMotif(angle, clockwise=clockwise)
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform,
    )

    center = (100.0, 50.0)
    pystiche_transform = transforms.RotateMotif(angle, center=center)
    pil_transform = PILRotateMotif(angle, center=center)
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform,
    )


def test_ScaleMotif():
    def PILScaleMotif(factor, center=None):
        if not isinstance(factor, float):
            factor = tuple(factor[::-1])
        if center is not None:
            center = tuple(center[::-1])
        affine_transform = pa.Scale(factor, center=center)
        return get_pil_affine_transform(affine_transform)

    factor = 2.0
    pystiche_transform = transforms.ScaleMotif(factor)
    pil_transform = PILScaleMotif(factor)
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform,
    )

    factor = (0.3, 0.7)
    pystiche_transform = transforms.ScaleMotif(factor)
    pil_transform = PILScaleMotif(factor)
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform,
    )

    factor = 0.5
    center = (100.0, 50.0)
    pystiche_transform = transforms.ScaleMotif(factor, center=center)
    pil_transform = PILScaleMotif(factor, center=center)
    pystiche_transform(torch.rand(1, 100, 100))
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform,
    )


def test_TranslateMotif():
    def PILTranslateMotif(translation, inverse=False):
        translation = tuple(translation[::-1])
        affine_transform = pa.Translate(translation, inverse=inverse)
        return get_pil_affine_transform(affine_transform)

    translation = (100.0, 100.0)
    pystiche_transform = transforms.TranslateMotif(translation)
    pil_transform = PILTranslateMotif(translation)
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform
    )

    inverse = True
    pystiche_transform = transforms.TranslateMotif(translation, inverse=inverse)
    pil_transform = PILTranslateMotif(translation, inverse=inverse)
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform
    )


def test_TransformMotifAffinely():
    def PILTransformMotifAffinely(
        shearing_angle=0.0,
        clockwise_shearing=False,
        shearing_center=None,
        rotation_angle=0.0,
        clockwise_rotation=False,
        rotation_center=None,
        scaling_factor=1.0,
        scaling_center=None,
        translation=(0.0, 0.0),
        inverse_translation=False,
    ):
        if shearing_center is not None:
            shearing_center = tuple(shearing_center[::-1])
        if rotation_center is not None:
            rotation_center = tuple(rotation_center[::-1])
        if scaling_center is not None:
            scaling_center = tuple(scaling_center[::-1])
        if not isinstance(scaling_factor, float):
            scaling_factor = tuple(scaling_factor[::-1])
        translation = tuple(translation[::-1])
        affine_transform = pa.ComposedTransform(
            pa.Shear(
                shearing_angle, clockwise=clockwise_shearing, center=shearing_center
            ),
            pa.Rotate(
                rotation_angle, clockwise=clockwise_rotation, center=rotation_center
            ),
            pa.Scale(scaling_factor, center=scaling_center),
            pa.Translate(translation, inverse=inverse_translation),
        )
        return get_pil_affine_transform(affine_transform)

    transform_kwargs = {
        "shearing_angle": 10.0,
        "clockwise_shearing": True,
        "shearing_center": None,
        "rotation_angle": 20.0,
        "clockwise_rotation": True,
        "rotation_center": (0.0, 0.0),
        "scaling_factor": (0.6, 0.7),
        "scaling_center": (300.0, 200.0),
        "translation": (-20.0, 50.0),
        "inverse_translation": False,
    }

    pystiche_transform = transforms.TransformMotifAffinely(**transform_kwargs)
    pil_transform = PILTransformMotifAffinely(**transform_kwargs)
    pystiche_transform(torch.rand(1, 100, 100))
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform
    )


def test_TransformMotifAffinely_empty():
    with pytest.raises(RuntimeError):
        transforms.TransformMotifAffinely()


def test_TransformMotifAffinely_full_canvas():
    def PILRotateMotif(angle, expand=False):
        affine_transform = pa.Rotate(angle)
        return get_pil_affine_transform(affine_transform, expand=expand)

    angle = 30.0
    pystiche_transform = transforms.RotateMotif(angle=angle, canvas="full")
    pil_transform = PILRotateMotif(angle=angle, expand=True)

    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform,
    )


def test_TransformMotifAffinely_valid_canvas():
    pystiche_image = torch.ones(1, 1, 100, 100)

    angle = 45.0
    canvas = "valid"
    pystiche_transform = transforms.RotateMotif(angle=angle, canvas=canvas)
    with pytest.raises(RuntimeError):
        pystiche_transform(pystiche_image)
