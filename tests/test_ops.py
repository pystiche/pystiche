import torch
from torch import nn
from torch.nn.functional import mse_loss

import pystiche
from pystiche import ops
from pystiche.enc import MultiLayerEncoder, SequentialEncoder, SingleLayerEncoder
from pystiche.ops import functional as F
from utils import PysticheTestCase


class TestComparison(PysticheTestCase):
    def test_MSEEncodingOperator_call(self):
        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 128, 128)
        input_image = torch.rand(1, 3, 128, 128)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

        op = ops.MSEEncodingOperator(encoder)
        op.set_target_image(target_image)

        actual = op(input_image)
        desired = mse_loss(encoder(input_image), encoder(target_image))
        self.assertTensorAlmostEqual(actual, desired)

    def test_GramOperator_call(self):
        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 128, 128)
        input_image = torch.rand(1, 3, 128, 128)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

        op = ops.GramOperator(encoder)
        op.set_target_image(target_image)

        actual = op(input_image)
        desired = mse_loss(
            pystiche.batch_gram_matrix(encoder(input_image), normalize=True),
            pystiche.batch_gram_matrix(encoder(target_image), normalize=True),
        )
        self.assertTensorAlmostEqual(actual, desired)

    def test_MRFOperator_call(self):
        patch_size = 3
        stride = 2

        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 128, 128)
        input_image = torch.rand(1, 3, 128, 128)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

        op = ops.MRFOperator(encoder, patch_size, stride=stride)
        op.set_target_image(target_image)

        actual = op(input_image)
        desired = F.patch_matching_loss(
            pystiche.extract_patches2d(encoder(input_image), patch_size, stride=stride),
            pystiche.extract_patches2d(
                encoder(target_image), patch_size, stride=stride
            ),
        )
        self.assertFloatAlmostEqual(actual, desired)


class TestContainer(PysticheTestCase):
    def test_OperatorContainer(self):
        class TestOperator(ops.Operator):
            def process_input_image(self, image):
                pass

        named_ops = [(str(idx), TestOperator()) for idx in range(3)]
        op_container = ops.OperatorContainer(named_ops)

        actuals = op_container.named_children()
        desireds = named_ops
        self.assertNamedChildrenEqual(actuals, desireds)

    def test_OperatorContainer_call(self):
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
        self.assertTensorDictAlmostEqual(actual, desired)

    def test_OperatorContainer_getitem(self):
        class TestOperator(ops.Operator):
            def __init__(self, bias):
                super().__init__()
                self.bias = bias

            def process_input_image(self, image):
                return image + self.bias

        named_ops = [(str(idx), TestOperator(idx + 1.0)) for idx in range(3)]
        op_container = ops.OperatorContainer(named_ops)

        for name, _ in named_ops:
            actual = op_container[name]
            desired = getattr(op_container, name)
            self.assertIs(actual, desired)

    def test_SameOperatorContainer(self):
        class TestOperator(ops.Operator):
            def process_input_image(self, image):
                pass

        def get_op(name, score_weight):
            return TestOperator()

        names = [str(idx) for idx in range(3)]
        same_operator_container = ops.SameOperatorContainer(names, get_op)

        for name in names:
            op = getattr(same_operator_container, name)
            self.assertIsInstance(op, TestOperator)

    def test_SameOperatorContainer_op_weights_str(self):
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
            self.assertFloatAlmostEqual(actual, desired)

        with self.assertRaises(ValueError):
            ops.SameOperatorContainer(
                names, get_op, op_weights="invalid",
            )

    def test_SameOperatorContainer_op_weights_seq(self):
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
            self.assertFloatAlmostEqual(actual, desired)

    def test_MultiLayerEncodingOperator(self):
        class TestOperator(ops.EncodingRegularizationOperator):
            def input_enc_to_repr(self, image):
                pass

            def calculate_score(self, input_repr):
                pass

        def get_encoding_op(encoder, score_weight):
            return TestOperator(encoder, score_weight=score_weight)

        layers = [str(index) for index in range(3)]
        modules = [(layer, nn.Module()) for layer in layers]
        multi_layer_encoder = MultiLayerEncoder(modules)

        multi_layer_enc_op = ops.MultiLayerEncodingOperator(
            multi_layer_encoder, layers, get_encoding_op
        )

        for layer in layers:
            op = getattr(multi_layer_enc_op, layer)
            self.assertIsInstance(op.encoder, SingleLayerEncoder)
            self.assertEqual(op.encoder.layer, layer)
            self.assertIs(op.encoder.multi_layer_encoder, multi_layer_encoder)

    def test_MultiLayerEncodingOperator_set_target_image(self):
        class TestOperator(ops.EncodingComparisonOperator):
            def target_enc_to_repr(self, image):
                return image, None

            def input_enc_to_repr(self, image, ctx):
                pass

            def calculate_score(self, input_repr, target_repr, ctx):
                pass

        def get_encoding_op(encoder, score_weight):
            return TestOperator(encoder, score_weight=score_weight)

        torch.manual_seed(0)
        image = torch.rand(1, 3, 128, 128)

        layers = [str(index) for index in range(3)]
        modules = [(layer, nn.Conv2d(3, 3, 1)) for layer in layers]
        multi_layer_encoder = MultiLayerEncoder(modules)

        multi_layer_enc_op = ops.MultiLayerEncodingOperator(
            multi_layer_encoder, layers, get_encoding_op
        )
        multi_layer_enc_op.set_target_image(image)

        for layer in layers:
            actual = getattr(multi_layer_enc_op, layer).target_image
            desired = image
            self.assertTensorAlmostEqual(actual, desired)

    def test_MultiRegionOperator_set_regional_target_image(self):
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

        torch.manual_seed(0)
        image = torch.rand(1, 3, 128, 128)

        regions = [str(idx) for idx in range(3)]
        multi_region_operator = ops.MultiRegionOperator(regions, get_op)
        for region in regions:
            multi_region_operator.set_regional_target_image(region, image)

        for region in regions:
            actual = getattr(multi_region_operator, region).target_image
            desired = image
            self.assertTensorAlmostEqual(actual, desired)


class TestFunctional(PysticheTestCase):
    def test_reduce(self):
        torch.manual_seed(0)
        x = torch.rand(1, 3, 128, 128)

        actual = F._reduce(x, "mean")
        desired = torch.mean(x)
        self.assertTensorAlmostEqual(actual, desired)

        actual = F._reduce(x, "sum")
        desired = torch.sum(x)
        self.assertTensorAlmostEqual(actual, desired)

        actual = F._reduce(x, "none")
        desired = x
        self.assertTensorAlmostEqual(actual, desired)

    def test_patch_matching_loss(self):
        torch.manual_seed(0)
        zero_patch = torch.zeros(3, 3, 3)
        one_patch = torch.ones(3, 3, 3)
        rand_patch = torch.randn(3, 3, 3)

        input = torch.stack((rand_patch + 0.1, rand_patch * 0.9))
        target = torch.stack((zero_patch, one_patch, rand_patch))

        actual = F.patch_matching_loss(input, target)
        desired = mse_loss(input, torch.stack((rand_patch, rand_patch)))
        self.assertFloatAlmostEqual(actual, desired)

        pass

    def test_value_range_loss_zero(self):
        torch.manual_seed(0)
        input = torch.rand(1, 3, 128, 128)

        actual = F.value_range_loss(input)
        desired = 0.0
        self.assertFloatAlmostEqual(actual, desired)

    def test_total_variation_loss(self):
        def get_checkerboard(size):
            return (
                (
                    torch.arange(size ** 2).view(size, size)
                    + torch.arange(size).view(size, 1)
                )
                % 2
            ).bool()

        size = 128
        checkerboard = get_checkerboard(size)
        input = checkerboard.float().view(1, 1, size, size).repeat(1, 3, 1, 1)

        actual = F.total_variation_loss(input)
        desired = 2.0
        self.assertFloatAlmostEqual(actual, desired)


class TestGuidance(PysticheTestCase):
    pass


class TestOp(PysticheTestCase):
    def test_Operator_call(self):
        class TestOperator(ops.Operator):
            def process_input_image(self, image):
                return image + 1.0

        torch.manual_seed(0)
        image = torch.rand(1, 3, 128, 128)

        test_op = TestOperator(score_weight=2.0)

        actual = test_op(image)
        desired = (image + 1.0) * 2.0
        self.assertTensorAlmostEqual(actual, desired)

    def test_PixelRegularizationOperator_call(self):
        class TestOperator(ops.PixelRegularizationOperator):
            def input_image_to_repr(self, image):
                return image * 2.0

            def calculate_score(self, input_repr):
                return input_repr + 1.0

        torch.manual_seed(0)
        image = torch.rand(1, 3, 128, 128)

        test_op = TestOperator()

        actual = test_op(image)
        desired = image * 2.0 + 1.0
        self.assertTensorAlmostEqual(actual, desired)

    def test_EncodingRegularizationOperator_call(self):
        class TestOperator(ops.EncodingRegularizationOperator):
            def input_enc_to_repr(self, image):
                return image * 2.0

            def calculate_score(self, input_repr):
                return input_repr + 1.0

        torch.manual_seed(0)
        image = torch.rand(1, 3, 128, 128)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

        test_op = TestOperator(encoder)

        actual = test_op(image)
        desired = encoder(image) * 2.0 + 1.0
        self.assertTensorAlmostEqual(actual, desired)

    def test_PixelComparisonOperator_set_target_image(self):
        class TestOperator(ops.PixelComparisonOperator):
            def target_image_to_repr(self, image):
                repr = image * 2.0
                ctx = torch.norm(image)
                return repr, ctx

            def input_image_to_repr(self, image, ctx):
                pass

            def calculate_score(self, input_repr, target_repr, ctx):
                pass

        torch.manual_seed(0)
        image = torch.rand(1, 3, 128, 128)

        test_op = TestOperator()
        self.assertFalse(test_op.has_target_image)

        test_op.set_target_image(image)
        self.assertTrue(test_op.has_target_image)

        actual = test_op.target_image
        desired = image
        self.assertTensorAlmostEqual(actual, desired)

        actual = test_op.target_repr
        desired = image * 2.0
        self.assertTensorAlmostEqual(actual, desired)

        actual = test_op.ctx
        desired = torch.norm(image)
        self.assertTensorAlmostEqual(actual, desired)

    def test_PixelComparisonOperator_call(self):
        class TestOperator(ops.PixelComparisonOperator):
            def target_image_to_repr(self, image):
                repr = image + 1.0
                return repr, None

            def input_image_to_repr(self, image, ctx):
                return image + 2.0

            def calculate_score(self, input_repr, target_repr, ctx):
                return input_repr * target_repr

        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 128, 128)
        input_image = torch.rand(1, 3, 128, 128)

        test_op = TestOperator()
        test_op.set_target_image(target_image)

        actual = test_op(input_image)
        desired = (target_image + 1.0) * (input_image + 2.0)
        self.assertTensorAlmostEqual(actual, desired)

    def test_PixelComparisonOperator_call_no_target(self):
        class TestOperator(ops.PixelComparisonOperator):
            def target_image_to_repr(self, image):
                pass

            def input_image_to_repr(self, image, ctx):
                pass

            def calculate_score(self, input_repr, target_repr, ctx):
                pass

        torch.manual_seed(0)
        input_image = torch.rand(1, 3, 128, 128)

        test_op = TestOperator()

        with self.assertRaises(RuntimeError):
            test_op(input_image)

    def test_EncodingComparisonOperator_set_target_image(self):
        class TestOperator(ops.EncodingComparisonOperator):
            def target_enc_to_repr(self, image):
                repr = image * 2.0
                ctx = torch.norm(image)
                return repr, ctx

            def input_enc_to_repr(self, image, ctx):
                pass

            def calculate_score(self, input_repr, target_repr, ctx):
                pass

        torch.manual_seed(0)
        image = torch.rand(1, 3, 128, 128)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

        test_op = TestOperator(encoder)
        self.assertFalse(test_op.has_target_image)

        test_op.set_target_image(image)
        self.assertTrue(test_op.has_target_image)

        actual = test_op.target_image
        desired = image
        self.assertTensorAlmostEqual(actual, desired)

        actual = test_op.target_repr
        desired = encoder(image) * 2.0
        self.assertTensorAlmostEqual(actual, desired)

        actual = test_op.ctx
        desired = torch.norm(encoder(image))
        self.assertTensorAlmostEqual(actual, desired)

    def test_EncodingComparisonOperator_call(self):
        class TestOperator(ops.EncodingComparisonOperator):
            def target_enc_to_repr(self, image):
                repr = image + 1.0
                return repr, None

            def input_enc_to_repr(self, image, ctx):
                return image + 2.0

            def calculate_score(self, input_repr, target_repr, ctx):
                return input_repr * target_repr

        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 128, 128)
        input_image = torch.rand(1, 3, 128, 128)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

        test_op = TestOperator(encoder)
        test_op.set_target_image(target_image)

        actual = test_op(input_image)
        desired = (encoder(target_image) + 1.0) * (encoder(input_image) + 2.0)
        self.assertTensorAlmostEqual(actual, desired)

    def test_EncodingComparisonOperator_call_no_target(self):
        class TestOperator(ops.EncodingComparisonOperator):
            def target_enc_to_repr(self, image):
                pass

            def input_enc_to_repr(self, image, ctx):
                pass

            def calculate_score(self, input_repr, target_repr, ctx):
                pass

        torch.manual_seed(0)
        input_image = torch.rand(1, 3, 128, 128)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

        test_op = TestOperator(encoder)

        with self.assertRaises(RuntimeError):
            test_op(input_image)


class TestRegularization(PysticheTestCase):
    def test_TotalVariationOperator_call(self):
        torch.manual_seed(0)
        image = torch.rand(1, 3, 128, 128)
        exponent = 3.0

        op = ops.TotalVariationOperator(exponent=exponent)

        actual = op(image)
        desired = F.total_variation_loss(image, exponent=exponent)
        self.assertTensorAlmostEqual(actual, desired)

    def test_ValueRangeOperator_call(self):
        torch.manual_seed(0)
        image = torch.randn(1, 3, 128, 128)

        op = ops.ValueRangeOperator()

        actual = op(image)
        desired = F.value_range_loss(image)
        self.assertTensorAlmostEqual(actual, desired)
