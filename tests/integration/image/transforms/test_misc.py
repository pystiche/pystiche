import pytorch_testing_utils as ptu

import torch

from pystiche.image import transforms

from . import assert_is_identity_transform


def test_FloatToUint8Range():
    image = torch.tensor(1.0)
    transform = transforms.FloatToUint8Range()

    actual = transform(image)
    desired = image * 255.0
    ptu.assert_allclose(actual, desired)


def test_Uint8ToFloatRange():
    image = torch.tensor(255.0)
    transform = transforms.Uint8ToFloatRange()

    actual = transform(image)
    desired = image / 255.0
    ptu.assert_allclose(actual, desired)


def test_FloatToUint8Range_Uint8ToFloatRange_identity():
    float_to_uint8_range = transforms.FloatToUint8Range()
    uint8_to_float_range = transforms.Uint8ToFloatRange()

    assert_is_identity_transform(
        lambda image: uint8_to_float_range(float_to_uint8_range(image))
    )
    assert_is_identity_transform(
        lambda image: float_to_uint8_range(uint8_to_float_range(image))
    )


def test_ReverseChannelOrder(test_image):
    transform = transforms.ReverseChannelOrder()

    actual = transform(test_image)
    desired = test_image.flip(1)
    ptu.assert_allclose(actual, desired)


def test_ReverseChannelOrder_identity():
    transform = transforms.ReverseChannelOrder()

    assert_is_identity_transform(lambda image: transform(transform(image)))


def test_Normalize():
    mean = (0.0, -1.0, 2.0)
    std = (1e0, 1e-1, 1e1)
    transform = transforms.Normalize(mean, std)

    torch.manual_seed(0)
    normalized_image = torch.randn((1, 3, 256, 256))

    def to_tensor(seq):
        return torch.tensor(seq).view(1, -1, 1, 1)

    image = normalized_image * to_tensor(std) + to_tensor(mean)

    actual = transform(image)
    desired = normalized_image
    ptu.assert_allclose(actual, desired, atol=1e-6)


def test_Denormalize():
    mean = (0.0, -1.0, 2.0)
    std = (1e0, 1e-1, 1e1)
    transform = transforms.Denormalize(mean, std)

    torch.manual_seed(0)
    normalized_image = torch.randn((1, 3, 256, 256))

    def to_tensor(seq):
        return torch.tensor(seq).view(1, -1, 1, 1)

    image = normalized_image * to_tensor(std) + to_tensor(mean)

    actual = transform(normalized_image)
    desired = image
    ptu.assert_allclose(actual, desired, atol=1e-6)


def test_Normalize_Denormalize_identity():
    mean = (0.0, -1.0, 2.0)
    std = (1e0, 1e-1, 1e1)

    normalize = transforms.Normalize(mean, std)
    denormalize = transforms.Denormalize(mean, std)

    assert_is_identity_transform(lambda image: denormalize(normalize(image)))
    assert_is_identity_transform(lambda image: normalize(denormalize(image)))
