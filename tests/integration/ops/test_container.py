import pytest
import pytorch_testing_utils as ptu

import torch
from torch import nn

import pystiche
from pystiche import enc, ops

from tests.asserts import assert_named_modules_identical


class TestOperatorContainer:
    def test_main(self):
        class TestOperator(ops.Operator):
            def process_input_image(self, image):
                pass

        named_ops = [(str(idx), TestOperator()) for idx in range(3)]
        op_container = ops.OperatorContainer(named_ops)

        actuals = op_container.named_children()
        desireds = named_ops
        assert_named_modules_identical(actuals, desireds)

    def test_get_image_or_guide(self, subtests):
        class RegularizationTestOperator(ops.PixelRegularizationOperator):
            def input_image_to_repr(self, image):
                pass

            def calculate_score(self, input_repr):
                pass

        class ComparisonTestOperator(ops.PixelComparisonOperator):
            def target_image_to_repr(self, image):
                return image, None

            def input_image_to_repr(self, image, ctx):
                pass

            def calculate_score(self, input_repr, target_repr, ctx):
                pass

        def get_container():
            return ops.OperatorContainer(
                (
                    ("regularization", RegularizationTestOperator()),
                    ("comparison", ComparisonTestOperator()),
                )
            )

        torch.manual_seed(0)
        image_or_guide = torch.rand(1, 3, 128, 128)

        with subtests.test("no image or guide"):
            container = get_container()

            with pytest.raises(RuntimeError):
                container.get_input_guide()

        with subtests.test("mismatching images or guides"):
            container = get_container()
            container.regularization.set_input_guide(image_or_guide)
            container.comparison.set_input_guide(-image_or_guide)

            with pytest.raises(RuntimeError):
                container.get_input_guide()

        with subtests.test("get_target_guide"):
            container = get_container()
            container.comparison.set_target_guide(image_or_guide)

            actual = container.get_target_guide()
            desired = image_or_guide
            ptu.assert_allclose(actual, desired)

        with subtests.test("get_target_image"):
            container = get_container()
            container.comparison.set_target_image(image_or_guide)

            actual = container.get_target_image()
            desired = image_or_guide
            ptu.assert_allclose(actual, desired)

        with subtests.test("get_input_guide"):
            container = get_container()
            container.regularization.set_input_guide(image_or_guide)
            container.comparison.set_input_guide(image_or_guide)

            actual = container.get_input_guide()
            desired = image_or_guide
            ptu.assert_allclose(actual, desired)

    def test_set_image_or_guide(self, subtests):
        class RegularizationTestOperator(ops.PixelRegularizationOperator):
            def input_image_to_repr(self, image):
                pass

            def calculate_score(self, input_repr):
                pass

        class ComparisonTestOperator(ops.PixelComparisonOperator):
            def target_image_to_repr(self, image):
                return image, None

            def input_image_to_repr(self, image, ctx):
                pass

            def calculate_score(self, input_repr, target_repr, ctx):
                pass

        def get_container():
            return ops.OperatorContainer(
                (
                    ("regularization", RegularizationTestOperator()),
                    ("comparison", ComparisonTestOperator()),
                )
            )

        torch.manual_seed(0)
        image_or_guide = torch.rand(1, 3, 128, 128)

        with subtests.test("set_target_guide"):
            container = get_container()
            container.set_target_guide(image_or_guide, recalc_repr=False)

            actual = container.comparison.target_guide
            desired = image_or_guide
            ptu.assert_allclose(actual, desired)

        with subtests.test("set_target_guide"):
            container = get_container()
            container.set_target_image(image_or_guide)

            actual = container.comparison.target_image
            desired = image_or_guide
            ptu.assert_allclose(actual, desired)

        with subtests.test("set_input_guide"):
            container = get_container()
            container.set_input_guide(image_or_guide)

            actual = container.regularization.input_guide
            desired = image_or_guide
            ptu.assert_allclose(actual, desired)

            actual = container.comparison.input_guide
            desired = image_or_guide
            ptu.assert_allclose(actual, desired)

    def test_call(self):
        class TestOperator(ops.Operator):
            def __init__(self, bias):
                super().__init__()
                self.bias = bias

            def process_input_image(self, image):
                return image + self.bias

        input = torch.tensor(0.0)

        named_ops = [(str(idx), TestOperator(idx + 1.0)) for idx in range(3)]
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
    def test_set_regional_image_or_guide(self, subtests):
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

        def get_multi_region_operator():
            return ops.MultiRegionOperator(regions, get_op)

        methods_and_desired_attrs = (
            ("set_regional_target_guide", "target_guide"),
            ("set_regional_target_image", "target_image"),
            ("set_regional_input_guide", "input_guide"),
        )

        for method, desired_attr in methods_and_desired_attrs:
            with subtests.test(method):
                multi_region_operator = get_multi_region_operator()

                for region, image_or_guide in regional_images_or_guides:
                    getattr(multi_region_operator, method)(region, image_or_guide)

                for region, image_or_guide in regional_images_or_guides:
                    actual = getattr(
                        getattr(multi_region_operator, region), desired_attr
                    )
                    desired = image_or_guide
                    ptu.assert_allclose(actual, desired)
