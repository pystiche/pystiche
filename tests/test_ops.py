import torch
from torch import nn
from torch.nn.functional import mse_loss
import pystiche
from pystiche.enc import SequentialEncoder
from pystiche import ops
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
        self.assertTensorAlmostEqual(actual, desired)


class TestContainer(PysticheTestCase):
    pass


class TestFunctional(PysticheTestCase):
    pass


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
