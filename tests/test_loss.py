from itertools import chain, combinations

import torch
from torch import nn

import pystiche
from pystiche import loss
from pystiche.enc import MultiLayerEncoder, SequentialEncoder
from pystiche.ops import (
    EncodingRegularizationOperator,
    MSEEncodingOperator,
    Operator,
    TotalVariationOperator,
)
from utils import ForwardPassCounter, PysticheTestCase


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
        class TestOperator(EncodingRegularizationOperator):
            def input_enc_to_repr(self, image):
                pass

            def calculate_score(self, input_repr):
                pass

        layers = [str(idx) for idx in range(3)]
        modules = [(layer, nn.Module()) for layer in layers]
        multi_layer_encoder = MultiLayerEncoder(modules)

        ops = (
            (
                "op",
                TestOperator(
                    multi_layer_encoder.extract_single_layer_encoder(layers[0])
                ),
            ),
        )
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
        class TestOperator(EncodingRegularizationOperator):
            def input_enc_to_repr(self, image):
                return image

            def calculate_score(self, input_repr):
                return torch.mean(input_repr)

        count = ForwardPassCounter()
        modules = (("count", count),)
        multi_layer_encoder = MultiLayerEncoder(modules)

        ops = [
            (
                str(idx),
                TestOperator(multi_layer_encoder.extract_single_layer_encoder("count")),
            )
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
    def test_perceptual_loss(self):
        op = TotalVariationOperator()
        required_components = {"content_loss", "style_loss"}
        all_components = {*required_components, "regularization"}
        for components in powerset(all_components):
            if not set(components).intersection(required_components):
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
        content_loss = MSEEncodingOperator(SequentialEncoder((nn.Conv2d(1, 1, 1),)))
        style_loss = MSEEncodingOperator(SequentialEncoder((nn.Conv2d(1, 1, 1),)))

        perceptual_loss = loss.PerceptualLoss(style_loss=style_loss)
        with self.assertRaises(RuntimeError):
            perceptual_loss.set_content_image(image)

        perceptual_loss = loss.PerceptualLoss(
            content_loss=content_loss, style_loss=style_loss
        )
        perceptual_loss.set_content_image(image)

        self.assertTrue(content_loss.has_target_image)

        actual = content_loss.target_image
        desired = image
        self.assertTensorAlmostEqual(actual, desired)

    def test_perceptual_loss_style_image(self):
        torch.manual_seed(0)
        image = torch.rand(1, 1, 100, 100)
        content_loss = MSEEncodingOperator(SequentialEncoder((nn.Conv2d(1, 1, 1),)))
        style_loss = MSEEncodingOperator(SequentialEncoder((nn.Conv2d(1, 1, 1),)))

        perceptual_loss = loss.PerceptualLoss(content_loss=content_loss)
        with self.assertRaises(RuntimeError):
            perceptual_loss.set_style_image(image)

        perceptual_loss = loss.PerceptualLoss(
            content_loss=content_loss, style_loss=style_loss
        )
        perceptual_loss.set_style_image(image)

        self.assertTrue(style_loss.has_target_image)

        actual = style_loss.target_image
        desired = image
        self.assertTensorAlmostEqual(actual, desired)
