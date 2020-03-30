from itertools import chain, combinations
import torch
from torch import nn
from pystiche.ops import TotalVariationOperator, MSEEncodingOperator
from pystiche import loss
from utils import PysticheTestCase


# copied from
# https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class TestCase(PysticheTestCase):
    def test_perceptual_loss_components(self):
        op = TotalVariationOperator()
        all_components = {"content_loss", "style_loss", "regularization"}
        for components in powerset(all_components):
            if not components:
                with self.assertRaises(RuntimeError):
                    loss.PerceptualLoss()
                continue

            perceptual_loss = loss.PerceptualLoss(
                **{component: op for component in components}
            )

            for component in components:
                self.assertTrue(getattr(perceptual_loss, f"has_{component}"))
                self.assertIs(getattr(perceptual_loss, component), op)

            for component in all_components - set(components):
                self.assertFalse(getattr(perceptual_loss, f"has_{component}"))

    def test_perceptual_loss_content_image(self):
        torch.manual_seed(0)
        image = torch.rand(1, 1, 100, 100)
        regularization = TotalVariationOperator()

        perceptual_loss = loss.PerceptualLoss(regularization=regularization)
        with self.assertRaises(RuntimeError):
            perceptual_loss.set_content_image(image)

        content_loss = MSEEncodingOperator(nn.Conv2d(1, 1, 1))
        perceptual_loss = loss.PerceptualLoss(
            content_loss=content_loss, regularization=regularization
        )
        perceptual_loss.set_content_image(image)

        self.assertTrue(content_loss.has_target_image)

        actual = content_loss.target_image
        desired = image
        self.assertTensorAlmostEqual(actual, desired)

    def test_perceptual_loss_style_image(self):
        torch.manual_seed(0)
        image = torch.rand(1, 1, 100, 100)
        regularization = TotalVariationOperator()

        perceptual_loss = loss.PerceptualLoss(regularization=regularization)
        with self.assertRaises(RuntimeError):
            perceptual_loss.set_style_image(image)

        style_loss = MSEEncodingOperator(nn.Conv2d(1, 1, 1))
        perceptual_loss = loss.PerceptualLoss(
            style_loss=style_loss, regularization=regularization
        )
        perceptual_loss.set_style_image(image)

        self.assertTrue(style_loss.has_target_image)

        actual = style_loss.target_image
        desired = image
        self.assertTensorAlmostEqual(actual, desired)
