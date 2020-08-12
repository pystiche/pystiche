import torch

import pystiche


def test_extract_patchesnd_num_patches(self):
    batch_size = 2
    size = 64
    patch_size = 3
    stride = 2

    num_patches_per_dim = int((size - patch_size) // stride + 1)
    torch.manual_seed(0)
    for dims, extract_patches in enumerate(
        (
            pystiche.extract_patches1d,
            pystiche.extract_patches2d,
            pystiche.extract_patches3d,
        ),
        1,
    ):
        x = torch.rand(batch_size, 1, *[size] * dims)
        patches = extract_patches(x, patch_size, stride=stride)

        actual = patches.size()[0]
        desired = batch_size * num_patches_per_dim ** dims
        self.assertEqual(actual, desired)


def test_extract_patches1d(self):
    batch_size = 2
    length = 9
    patch_size = 3
    stride = 2

    x = torch.arange(batch_size * length).view(batch_size, 1, -1)
    patches = pystiche.extract_patches1d(x, patch_size, stride=stride)

    actual = patches[0]
    desired = x[0, :, :patch_size]
    self.assertTensorAlmostEqual(actual, desired)

    actual = patches[1]
    desired = x[0, :, stride : (stride + patch_size)]
    self.assertTensorAlmostEqual(actual, desired)

    actual = patches[-1]
    desired = x[-1, :, -patch_size:]
    self.assertTensorAlmostEqual(actual, desired)
