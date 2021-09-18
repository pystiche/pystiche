import pytorch_testing_utils as ptu

import torch

import pystiche
import pystiche.loss.functional as F
from pystiche import loss as loss_
from pystiche.misc import suppress_depr_warnings


class TestFeatureReconstructionLoss:
    @suppress_depr_warnings
    def test_call(self, encoder):
        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 128, 128)
        input_image = torch.rand(1, 3, 128, 128)

        loss = loss_.FeatureReconstructionLoss(encoder)
        loss.set_target_image(target_image)

        actual = loss(input_image)
        desired = F.mse_loss(encoder(input_image), encoder(target_image))
        ptu.assert_allclose(actual, desired)

    @suppress_depr_warnings
    def test_repr_smoke(self, encoder):
        assert isinstance(repr(loss_.FeatureReconstructionLoss(encoder)), str)


class TestGramLoss:
    @suppress_depr_warnings
    def test_call(self, encoder):
        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 128, 128)
        input_image = torch.rand(1, 3, 128, 128)

        loss = loss_.GramLoss(encoder)
        loss.set_target_image(target_image)

        actual = loss(input_image)
        desired = F.mse_loss(
            pystiche.gram_matrix(encoder(input_image), normalize=loss.normalize),
            pystiche.gram_matrix(encoder(target_image), normalize=loss.normalize),
        )
        ptu.assert_allclose(actual, desired)

    @suppress_depr_warnings
    def test_repr_smoke(self, encoder):
        assert isinstance(repr(loss_.GramLoss(encoder)), str)


class TestMRFLoss:
    @suppress_depr_warnings
    def test_call(self, encoder):
        patch_size = 3
        stride = 2

        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 32, 32)
        input_image = torch.rand(1, 3, 32, 32)

        loss = loss_.MRFLoss(encoder, patch_size, stride=stride)
        loss.set_target_image(target_image)

        actual = loss(input_image)
        desired = F.mrf_loss(
            pystiche.extract_patches2d(encoder(input_image), patch_size, stride=stride),
            pystiche.extract_patches2d(
                encoder(target_image), patch_size, stride=stride
            ),
        )
        ptu.assert_allclose(actual, desired)

    @suppress_depr_warnings
    def test_call_guided(self, encoder):
        patch_size = 2
        stride = 2

        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 32, 32)
        input_image = torch.rand(1, 3, 32, 32)
        target_guide = torch.cat(
            (torch.zeros(1, 1, 16, 32), torch.ones(1, 1, 16, 32)), dim=2
        )
        input_guide = target_guide.flip(2)

        loss = loss_.MRFLoss(encoder, patch_size, stride=stride)
        loss.set_target_image(target_image, guide=target_guide)
        loss.set_input_guide(input_guide)

        actual = loss(input_image)

        input_enc = encoder(input_image)[:, :, :16, :]
        target_enc = encoder(target_image)[:, :, 16:, :]
        desired = F.mrf_loss(
            pystiche.extract_patches2d(input_enc, patch_size, stride=stride),
            pystiche.extract_patches2d(target_enc, patch_size, stride=stride),
        )
        ptu.assert_allclose(actual, desired)

    @suppress_depr_warnings
    def test_repr_smoke(self, encoder):
        assert isinstance(repr(loss_.MRFLoss(encoder, 2, stride=1)), str)
