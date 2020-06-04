from itertools import chain, combinations

import torch
from torch import nn

import pystiche
from pystiche import loss
from pystiche.enc import MultiLayerEncoder, SequentialEncoder
from pystiche.ops import (
    EncodingOperator,
    FeatureReconstructionOperator,
    MultiRegionOperator,
    Operator,
    PixelComparisonOperator,
)

from .utils import ForwardPassCounter, PysticheTestCase


# copied from
# https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class TestMultiOp(PysticheTestCase):
    def test_MultiOperatorLoss(self):
        class TestOperator(Operator):
            def process_input_image(self, image):
                pass

        named_ops = [(str(idx), TestOperator()) for idx in range(3)]
        multi_op_loss = loss.MultiOperatorLoss(named_ops)

        actuals = multi_op_loss.named_children()
        desireds = named_ops
        self.assertNamedChildrenEqual(actuals, desireds)

    def test_MultiOperatorLoss_trim(self):
        class TestOperator(EncodingOperator):
            def __init__(self, encoder, **kwargs):
                super().__init__(**kwargs)
                self._encoder = encoder

            @property
            def encoder(self):
                return self._encoder

            def process_input_image(self, image):
                pass

        layers = [str(idx) for idx in range(3)]
        modules = [(layer, nn.Module()) for layer in layers]
        multi_layer_encoder = MultiLayerEncoder(modules)

        ops = (("op", TestOperator(multi_layer_encoder.extract_encoder(layers[0])),),)
        loss.MultiOperatorLoss(ops, trim=True)

        self.assertTrue(layers[0] in multi_layer_encoder)
        for layer in layers[1:]:
            self.assertFalse(layer in multi_layer_encoder)

    def test_MultiOperatorLoss_call(self):
        class TestOperator(Operator):
            def __init__(self, bias):
                super().__init__()
                self.bias = bias

            def process_input_image(self, image):
                return image + self.bias

        input = torch.tensor(0.0)

        named_ops = [(str(idx), TestOperator(idx + 1.0)) for idx in range(3)]
        multi_op_loss = loss.MultiOperatorLoss(named_ops)

        actual = multi_op_loss(input)
        desired = pystiche.LossDict([(name, input + op.bias) for name, op in named_ops])
        self.assertTensorDictAlmostEqual(actual, desired)

    def test_MultiOperatorLoss_call_encode(self):
        class TestOperator(EncodingOperator):
            def __init__(self, encoder, **kwargs):
                super().__init__(**kwargs)
                self._encoder = encoder

            @property
            def encoder(self):
                return self._encoder

            def process_input_image(self, image):
                return torch.sum(image)

        count = ForwardPassCounter()
        modules = (("count", count),)
        multi_layer_encoder = MultiLayerEncoder(modules)

        ops = [
            (str(idx), TestOperator(multi_layer_encoder.extract_encoder("count")),)
            for idx in range(3)
        ]
        multi_op_loss = loss.MultiOperatorLoss(ops)

        torch.manual_seed(0)
        input = torch.rand(1, 3, 128, 128)

        multi_op_loss(input)
        actual = count.count
        desired = 1
        self.assertEqual(actual, desired)

        multi_op_loss(input)
        actual = count.count
        desired = 2
        self.assertEqual(actual, desired)


class TestPerceptual(PysticheTestCase):
    def test_PerceptualLoss_set_content_image(self):
        torch.manual_seed(0)
        image = torch.rand(1, 1, 100, 100)
        content_loss = FeatureReconstructionOperator(
            SequentialEncoder((nn.Conv2d(1, 1, 1),))
        )
        style_loss = FeatureReconstructionOperator(
            SequentialEncoder((nn.Conv2d(1, 1, 1),))
        )

        perceptual_loss = loss.PerceptualLoss(content_loss, style_loss)
        perceptual_loss.set_content_image(image)

        actual = content_loss.target_image
        desired = image
        self.assertTensorAlmostEqual(actual, desired)

    def test_PerceptualLoss_set_style_image(self):
        torch.manual_seed(0)
        image = torch.rand(1, 1, 100, 100)
        content_loss = FeatureReconstructionOperator(
            SequentialEncoder((nn.Conv2d(1, 1, 1),))
        )
        style_loss = FeatureReconstructionOperator(
            SequentialEncoder((nn.Conv2d(1, 1, 1),))
        )

        perceptual_loss = loss.PerceptualLoss(content_loss, style_loss)
        perceptual_loss.set_style_image(image)

        actual = style_loss.target_image
        desired = image
        self.assertTensorAlmostEqual(actual, desired)

    def test_GuidedPerceptualLoss(self):
        class TestOperator(PixelComparisonOperator):
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
            content_loss = FeatureReconstructionOperator(
                SequentialEncoder((nn.Conv2d(1, 1, 1),))
            )
            style_loss = MultiRegionOperator(regions, get_op)
            return loss.GuidedPerceptualLoss(content_loss, style_loss)

        method_names_and_desired_attrs = (
            ("set_style_guide", "target_guide"),
            ("set_style_image", "target_image"),
            ("set_content_guide", "input_guide"),
        )

        for method_name, desired_attr in method_names_and_desired_attrs:
            with self.subTest(method_name):
                guided_perceptual_loss = get_guided_perceptual_loss()

                for region, image_or_guide in regional_images_or_guides:
                    method = getattr(guided_perceptual_loss, method_name)
                    method(region, image_or_guide)

                for region, image_or_guide in regional_images_or_guides:
                    actual = getattr(
                        getattr(guided_perceptual_loss.style_loss, region), desired_attr
                    )
                    desired = image_or_guide
                    self.assertTensorAlmostEqual(actual, desired)
