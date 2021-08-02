import pytest
import pytorch_testing_utils as ptu

import torch

import pystiche

import tests.utils


@pytest.mark.parametrize(
    ("extract_fn", "dims"),
    [
        pytest.param(extract_fn, dims, id=tests.utils.extract_fn_name(extract_fn))
        for dims, extract_fn in enumerate(
            (
                pystiche.extract_patches1d,
                pystiche.extract_patches2d,
                pystiche.extract_patches3d,
            ),
            1,
        )
    ],
)
def test_extract_patchesnd_num_patches(extract_fn, dims):
    batch_size = 2
    num_channels = 1
    size = 64
    patch_size = 3
    stride = 2

    num_patches_per_dim = int((size - patch_size) // stride + 1)
    torch.manual_seed(0)
    x = torch.rand(batch_size, num_channels, *[size] * dims)

    patches = extract_fn(x, patch_size, stride=stride)

    assert patches.shape == (
        batch_size,
        num_patches_per_dim ** dims,
        num_channels,
        *(patch_size,) * dims,
    )


def test_extract_patches1d():
    batch_size = 2
    num_channels = 1
    length = 9
    patch_size = 3
    stride = 2

    x = torch.arange(batch_size * num_channels * length).view(
        batch_size, num_channels, -1
    )
    patches = pystiche.extract_patches1d(x, patch_size, stride=stride)

    actual = patches[:, 0]
    desired = x[..., :patch_size]
    ptu.assert_allclose(actual, desired)

    actual = patches[:, 1]
    desired = x[..., stride : (stride + patch_size)]
    ptu.assert_allclose(actual, desired)

    actual = patches[:, -1]
    desired = x[..., -patch_size:]
    ptu.assert_allclose(actual, desired)
