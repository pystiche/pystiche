import pytorch_testing_utils as ptu

import torch
from torch import nn
from torch.nn.functional import mse_loss

import pystiche
import pystiche.ops.functional as F
from pystiche import enc, ops
from pystiche.misc import suppress_depr_warnings


@suppress_depr_warnings
def test_FeatureReconstructionOperator_call():
    torch.manual_seed(0)
    target_image = torch.rand(1, 3, 128, 128)
    input_image = torch.rand(1, 3, 128, 128)
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    op = ops.FeatureReconstructionOperator(encoder)
    op.set_target_image(target_image)

    actual = op(input_image)
    desired = mse_loss(encoder(input_image), encoder(target_image))
    ptu.assert_allclose(actual, desired)


@suppress_depr_warnings
def test_GramOperator_call():
    torch.manual_seed(0)
    target_image = torch.rand(1, 3, 128, 128)
    input_image = torch.rand(1, 3, 128, 128)
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    op = ops.GramOperator(encoder)
    op.set_target_image(target_image)

    actual = op(input_image)
    desired = mse_loss(
        pystiche.gram_matrix(encoder(input_image), normalize=True),
        pystiche.gram_matrix(encoder(target_image), normalize=True),
    )
    ptu.assert_allclose(actual, desired)


@suppress_depr_warnings
def test_MRFOperator_scale_and_rotate_transforms_smoke():
    num_scale_steps = 2
    num_rotate_steps = 3

    target_transforms = ops.MRFOperator.scale_and_rotate_transforms(
        num_scale_steps=num_scale_steps, num_rotate_steps=num_rotate_steps,
    )

    actual = len(target_transforms)
    expected = (num_scale_steps * 2 + 1) * (num_rotate_steps * 2 + 1)
    assert actual == expected


@suppress_depr_warnings
def test_MRFOperator_enc_to_repr_guided(subtests):
    class Identity(pystiche.Module):
        def forward(self, image):
            return image

    patch_size = 2
    stride = 2

    op = ops.MRFOperator(
        enc.SequentialEncoder((Identity(),)), patch_size, stride=stride
    )

    with subtests.test(enc="constant"):
        enc_ = torch.ones(1, 4, 8, 8)

        actual = op.enc_to_repr(enc_, is_guided=True)
        desired = torch.ones(1, 0, 4, stride, stride)
        ptu.assert_allclose(actual, desired)

    with subtests.test(enc="spatial_mix"):
        constant = torch.ones(1, 4, 4, 8)
        varying = torch.rand(1, 4, 4, 8)
        enc_ = torch.cat((constant, varying), dim=2)

        actual = op.enc_to_repr(enc_, is_guided=True)
        desired = pystiche.extract_patches2d(varying, patch_size, stride=stride)
        ptu.assert_allclose(actual, desired)

    with subtests.test(enc="channel_mix"):
        constant = torch.ones(1, 2, 8, 8)
        varying = torch.rand(1, 2, 8, 8)
        enc_ = torch.cat((constant, varying), dim=1)

        actual = op.enc_to_repr(enc_, is_guided=True)
        desired = pystiche.extract_patches2d(enc_, patch_size, stride=stride)
        ptu.assert_allclose(actual, desired)

    with subtests.test(enc="varying"):
        enc_ = torch.rand(1, 4, 8, 8)

        actual = op.enc_to_repr(enc_, is_guided=True)
        desired = pystiche.extract_patches2d(enc_, patch_size, stride=stride)
        ptu.assert_allclose(actual, desired)


@suppress_depr_warnings
def test_MRFOperator_set_target_guide():
    patch_size = 3
    stride = 2

    torch.manual_seed(0)
    image = torch.rand(1, 3, 32, 32)
    guide = torch.rand(1, 1, 32, 32)
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    op = ops.MRFOperator(encoder, patch_size, stride=stride)
    op.set_target_image(image)
    assert not op.has_target_guide

    op.set_target_guide(guide)
    assert op.has_target_guide

    actual = op.target_guide
    desired = guide
    ptu.assert_allclose(actual, desired)

    actual = op.target_image
    desired = image
    ptu.assert_allclose(actual, desired)


@suppress_depr_warnings
def test_MRFOperator_set_target_guide_without_recalc():
    patch_size = 3
    stride = 2

    torch.manual_seed(0)
    image = torch.rand(1, 3, 32, 32)
    guide = torch.rand(1, 1, 32, 32)
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    op = ops.MRFOperator(encoder, patch_size, stride=stride)
    op.set_target_image(image)
    desired = op.target_repr.clone()

    op.set_target_guide(guide, recalc_repr=False)
    actual = op.target_repr

    ptu.assert_allclose(actual, desired)


@suppress_depr_warnings
def test_MRFOperator_call():
    patch_size = 3
    stride = 2

    torch.manual_seed(0)
    target_image = torch.rand(1, 3, 32, 32)
    input_image = torch.rand(1, 3, 32, 32)
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    op = ops.MRFOperator(encoder, patch_size, stride=stride)
    op.set_target_image(target_image)

    actual = op(input_image)
    desired = F.mrf_loss(
        pystiche.extract_patches2d(encoder(input_image), patch_size, stride=stride),
        pystiche.extract_patches2d(encoder(target_image), patch_size, stride=stride),
    )
    ptu.assert_allclose(actual, desired)


@suppress_depr_warnings
def test_MRFOperator_call_guided():
    patch_size = 2
    stride = 2

    torch.manual_seed(0)
    target_image = torch.rand(1, 3, 32, 32)
    input_image = torch.rand(1, 3, 32, 32)
    target_guide = torch.cat(
        (torch.zeros(1, 1, 16, 32), torch.ones(1, 1, 16, 32)), dim=2
    )
    input_guide = target_guide.flip(2)
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    op = ops.MRFOperator(encoder, patch_size, stride=stride)
    op.set_target_guide(target_guide)
    op.set_target_image(target_image)
    op.set_input_guide(input_guide)

    actual = op(input_image)

    input_enc = encoder(input_image)[:, :, :16, :]
    target_enc = encoder(target_image)[:, :, 16:, :]
    desired = F.mrf_loss(
        pystiche.extract_patches2d(input_enc, patch_size, stride=stride),
        pystiche.extract_patches2d(target_enc, patch_size, stride=stride),
    )
    ptu.assert_allclose(actual, desired)
