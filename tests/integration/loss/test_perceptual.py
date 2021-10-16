import pytest
import pytorch_testing_utils as ptu

import torch
from torch import nn

from pystiche import enc, loss, ops


class TestPerceptualLoss:
    def test_set_content_image(self):
        torch.manual_seed(0)
        image = torch.rand(1, 1, 100, 100)
        content_loss = ops.FeatureReconstructionLoss(
            enc.SequentialEncoder((nn.Conv2d(1, 1, 1),))
        )
        style_loss = ops.FeatureReconstructionLoss(
            enc.SequentialEncoder((nn.Conv2d(1, 1, 1),))
        )

        perceptual_loss = loss.PerceptualLoss(content_loss, style_loss)
        perceptual_loss.set_content_image(image)

        actual = content_loss.target_image
        desired = image
        ptu.assert_allclose(actual, desired)

    def test_set_style_image(self):
        torch.manual_seed(0)
        image = torch.rand(1, 1, 100, 100)
        content_loss = ops.FeatureReconstructionLoss(
            enc.SequentialEncoder((nn.Conv2d(1, 1, 1),))
        )
        style_loss = ops.FeatureReconstructionLoss(
            enc.SequentialEncoder((nn.Conv2d(1, 1, 1),))
        )

        perceptual_loss = loss.PerceptualLoss(content_loss, style_loss)
        perceptual_loss.set_style_image(image)

        actual = style_loss.target_image
        desired = image
        ptu.assert_allclose(actual, desired)


@pytest.mark.parametrize(
    ("method_name", "desired_attr"),
    (
        pytest.param("set_style_guide", "target_guide", id="style_guide"),
        pytest.param("set_style_image", "target_image", id="style_image"),
        pytest.param("set_content_guide", "input_guide", id="content_guide"),
    ),
)
def test_GuidedPerceptualLoss(method_name, desired_attr):
    class TestOperator(ops.PixelComparisonOperator):
        def __init__(self, bias, score_weight=1e0):
            super().__init__(score_weight=score_weight)
            self.bias = bias

        def target_image_to_repr(self, image):
            return image + self.bias, None

        def input_image_to_repr(self, image, ctx):
            pass

        def calculate_score(self, input_repr, target_repr, ctx):
            pass

    def get_op(name, score_weight):
        return TestOperator(float(name), score_weight=score_weight)

    regions = [str(idx) for idx in range(3)]
    torch.manual_seed(0)
    regional_images_or_guides = [
        (region, torch.rand(1, 3, 128, 128)) for region in regions
    ]

    def get_guided_perceptual_loss():
        content_loss = ops.FeatureReconstructionLoss(
            enc.SequentialEncoder((nn.Conv2d(1, 1, 1),))
        )
        style_loss = ops.MultiRegionOperator(regions, get_op)
        return loss.GuidedPerceptualLoss(content_loss, style_loss)

    guided_perceptual_loss = get_guided_perceptual_loss()

    for region, image_or_guide in regional_images_or_guides:
        method = getattr(guided_perceptual_loss, method_name)
        method(region, image_or_guide)

    for region, image_or_guide in regional_images_or_guides:
        actual = getattr(
            getattr(guided_perceptual_loss.style_loss, region), desired_attr
        )
        desired = image_or_guide
        ptu.assert_allclose(actual, desired)
