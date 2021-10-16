import pytest
import pytorch_testing_utils as ptu

import torch
from torch import nn

import pystiche
from pystiche import enc, ops

from tests.asserts import assert_named_modules_identical


class TestOperatorContainer:
    class Operator(ops.Operator):
        def __init__(self, bias=None):
            super().__init__()
            self.bias = bias

        def process_input_image(self, image):
            if not self.bias:
                return image

            return image + self.bias

    class RegularizationOperator(ops.PixelRegularizationOperator):
        def input_image_to_repr(self, image):
            pass

        def calculate_score(self, input_repr):
            pass

    class ComparisonOperator(ops.PixelComparisonOperator):
        def target_image_to_repr(self, image):
            return image, None

        def input_image_to_repr(self, image, ctx):
            pass

        def calculate_score(self, input_repr, target_repr, ctx):
            pass

    @pytest.fixture
    def op_container(self):
        return ops.OperatorContainer(
            (
                ("regularization", self.RegularizationOperator()),
                ("comparison", self.ComparisonOperator()),
            )
        )

    def test_main(self):
        named_ops = [(str(idx), self.Operator()) for idx in range(3)]
        op_container = ops.OperatorContainer(named_ops)

        actuals = op_container.named_children()
        desireds = named_ops
        assert_named_modules_identical(actuals, desireds)

    def test_get_image_or_guide_no(self, op_container):
        with pytest.raises(RuntimeError):
            op_container.get_input_guide()

    def test_get_image_or_guide_mismatching(self, test_image, op_container):
        op_container.regularization.set_input_guide(test_image)
        op_container.comparison.set_input_guide(-test_image)

        with pytest.raises(RuntimeError):
            op_container.get_input_guide()

    def test_get_image_or_guide_target_guide(self, test_image, op_container):
        op_container.comparison.set_target_guide(test_image)
        ptu.assert_allclose(op_container.get_target_guide(), test_image)

    def test_get_image_or_guide_target_image(self, test_image, op_container):
        op_container.comparison.set_target_image(test_image)
        ptu.assert_allclose(op_container.get_target_image(), test_image)

    def test_get_image_or_guide_input_guide(self, test_image, op_container):
        op_container.comparison.set_input_guide(test_image)
        ptu.assert_allclose(op_container.get_input_guide(), test_image)

    def test_set_image_or_guide_target_guide(self, test_image, op_container):
        op_container.set_target_guide(test_image, recalc_repr=False)
        ptu.assert_allclose(op_container.comparison.target_guide, test_image)

    def test_set_image_or_guide_target_image(self, test_image, op_container):
        op_container.set_target_image(test_image)
        ptu.assert_allclose(op_container.comparison.target_image, test_image)

    def test_set_image_or_guide_input_guide(self, test_image, op_container):
        op_container.set_input_guide(test_image)
        ptu.assert_allclose(op_container.comparison.input_guide, test_image)

    def test_call(self):
        input = torch.tensor(0.0)

        named_ops = [(str(idx), self.Operator(idx + 1.0)) for idx in range(3)]
        op_container = ops.OperatorContainer(named_ops)

        actual = op_container(input)
        desired = pystiche.LossDict([(name, input + op.bias) for name, op in named_ops])
        ptu.assert_allclose(actual, desired)


class TestSameOperatorContainer:
    def test_main(self):
        class TestOperator(ops.Operator):
            def process_input_image(self, image):
                pass

        def get_op(name, score_weight):
            return TestOperator()

        names = [str(idx) for idx in range(3)]
        same_operator_container = ops.SameOperatorContainer(names, get_op)

        for name in names:
            op = getattr(same_operator_container, name)
            assert isinstance(op, TestOperator)

    def test_op_weights_str(self):
        class TestOperator(ops.Operator):
            def process_input_image(self, image):
                pass

        def get_op(name, score_weight):
            return TestOperator(score_weight=score_weight)

        names = [str(idx) for idx in range(3)]
        op_weights_config = ("sum", "mean")

        desireds = (float(len(names)), 1.0)
        for op_weights, desired in zip(op_weights_config, desireds):
            same_operator_container = ops.SameOperatorContainer(
                names, get_op, op_weights=op_weights
            )
            actual = sum(
                [getattr(same_operator_container, name).score_weight for name in names]
            )
            assert actual == ptu.approx(desired)

        with pytest.raises(ValueError):
            ops.SameOperatorContainer(
                names, get_op, op_weights="invalid",
            )

    def test_op_weights_seq(self):
        class TestOperator(ops.Operator):
            def process_input_image(self, image):
                pass

        def get_op(name, score_weight):
            return TestOperator(score_weight=score_weight)

        names, op_weights = zip(*[(str(idx), float(idx) + 1.0) for idx in range(3)])

        same_operator_container = ops.SameOperatorContainer(
            names, get_op, op_weights=op_weights
        )

        for name, score_weight in zip(names, op_weights):
            actual = getattr(same_operator_container, name).score_weight
            desired = score_weight
            assert actual == ptu.approx(desired)


class TestMultiLayerEncodingOperator:
    def test_main(self):
        class TestOperator(ops.EncodingRegularizationOperator):
            def input_enc_to_repr(self, image):
                pass

            def calculate_score(self, input_repr):
                pass

        def get_encoding_op(encoder, score_weight):
            return TestOperator(encoder, score_weight=score_weight)

        layers = [str(index) for index in range(3)]
        modules = [(layer, nn.Module()) for layer in layers]
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        multi_layer_enc_op = ops.MultiLayerEncodingOperator(
            multi_layer_encoder, layers, get_encoding_op
        )

        for layer in layers:
            op = getattr(multi_layer_enc_op, layer)
            assert isinstance(op.encoder, enc.SingleLayerEncoder)
            assert op.encoder.layer == layer
            assert op.encoder.multi_layer_encoder is multi_layer_encoder


class TestMultiRegionOperator:
    class Operator(ops.PixelComparisonOperator):
        def __init__(self, bias, score_weight=1e0):
            super().__init__(score_weight=score_weight)
            self.bias = bias

        def target_image_to_repr(self, image):
            return image + self.bias, None

        def input_image_to_repr(self, image, ctx):
            pass

        def calculate_score(self, input_repr, target_repr, ctx):
            pass

    def get_op(self, name, score_weight):
        return self.Operator(float(name), score_weight=score_weight)

    @pytest.mark.parametrize(
        ("method", "desired_attr"),
        (
            pytest.param(
                "set_regional_target_guide", "target_guide", id="target_guide"
            ),
            pytest.param(
                "set_regional_target_image", "target_image", id="target_image"
            ),
            pytest.param("set_regional_input_guide", "input_guide", id="input_guide"),
        ),
    )
    def test_set_regional_image_or_guide(self, method, desired_attr):
        regions = [str(idx) for idx in range(3)]
        torch.manual_seed(0)
        regional_images_or_guides = [
            (region, torch.rand(1, 3, 128, 128)) for region in regions
        ]

        multi_region_operator = ops.MultiRegionOperator(regions, self.get_op)

        for region, image_or_guide in regional_images_or_guides:
            getattr(multi_region_operator, method)(region, image_or_guide)

        for region, image_or_guide in regional_images_or_guides:
            actual = getattr(getattr(multi_region_operator, region), desired_attr)
            desired = image_or_guide
            ptu.assert_allclose(actual, desired)
