import itertools

import torch
from torch import nn
from torch.nn.functional import mse_loss

import pystiche
from pystiche import ops
from pystiche.enc import MultiLayerEncoder, SequentialEncoder, SingleLayerEncoder
from pystiche.ops import functional as F

from .utils import PysticheTestCase


class TestComparison(PysticheTestCase):
    def test_FeatureReconstructionOperator_call(self):
        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 128, 128)
        input_image = torch.rand(1, 3, 128, 128)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

        op = ops.FeatureReconstructionOperator(encoder)
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
            pystiche.gram_matrix(encoder(input_image), normalize=True),
            pystiche.gram_matrix(encoder(target_image), normalize=True),
        )
        self.assertTensorAlmostEqual(actual, desired)

    def test_MRFOperator_scale_and_rotate_transforms(self):
        num_scale_steps = 1
        scale_step_width = 10e-2
        num_rotate_steps = 1
        rotate_step_width = 30.0

        target_transforms = ops.MRFOperator.scale_and_rotate_transforms(
            num_scale_steps=num_scale_steps,
            scale_step_width=scale_step_width,
            num_rotate_steps=num_rotate_steps,
            rotate_step_width=rotate_step_width,
        )
        self.assertEqual(
            len(target_transforms),
            (num_scale_steps * 2 + 1) * (num_rotate_steps * 2 + 1),
        )

        actual = {
            (transform.scaling_factor, transform.rotation_angle)
            for transform in target_transforms
        }
        desired = set(
            itertools.product(
                (1.0 - scale_step_width, 1.0, 1.0 + scale_step_width),
                (-rotate_step_width, 0.0, rotate_step_width),
            )
        )
        self.assertSetEqual(actual, desired)

    def test_MRFOperator_enc_to_repr_guided(self):
        class Identity(pystiche.Module):
            def forward(self, image):
                return image

        patch_size = 2
        stride = 2

        op = ops.MRFOperator(
            SequentialEncoder((Identity(),)), patch_size, stride=stride
        )

        with self.subTest(enc="constant"):
            enc = torch.ones(1, 4, 8, 8)

            actual = op.enc_to_repr(enc, is_guided=True)
            desired = torch.ones(0, 4, stride, stride)
            self.assertTensorAlmostEqual(actual, desired)

        with self.subTest(enc="spatial_mix"):
            constant = torch.ones(1, 4, 4, 8)
            varying = torch.rand(1, 4, 4, 8)
            enc = torch.cat((constant, varying), dim=2)

            actual = op.enc_to_repr(enc, is_guided=True)
            desired = pystiche.extract_patches2d(varying, patch_size, stride=stride)
            self.assertTensorAlmostEqual(actual, desired)

        with self.subTest(enc="channel_mix"):
            constant = torch.ones(1, 2, 8, 8)
            varying = torch.rand(1, 2, 8, 8)
            enc = torch.cat((constant, varying), dim=1)

            actual = op.enc_to_repr(enc, is_guided=True)
            desired = pystiche.extract_patches2d(enc, patch_size, stride=stride)
            self.assertTensorAlmostEqual(actual, desired)

        with self.subTest(enc="varying"):
            enc = torch.rand(1, 4, 8, 8)

            actual = op.enc_to_repr(enc, is_guided=True)
            desired = pystiche.extract_patches2d(enc, patch_size, stride=stride)
            self.assertTensorAlmostEqual(actual, desired)

    def test_MRFOperator_set_target_guide(self):
        patch_size = 3
        stride = 2

        torch.manual_seed(0)
        image = torch.rand(1, 3, 32, 32)
        guide = torch.rand(1, 1, 32, 32)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

        op = ops.MRFOperator(encoder, patch_size, stride=stride)
        op.set_target_image(image)
        self.assertFalse(op.has_target_guide)

        op.set_target_guide(guide)
        self.assertTrue(op.has_target_guide)

        actual = op.target_guide
        desired = guide
        self.assertTensorAlmostEqual(actual, desired)

        actual = op.target_image
        desired = image
        self.assertTensorAlmostEqual(actual, desired)

    def test_MRFOperator_set_target_guide_without_recalc(self):
        patch_size = 3
        stride = 2

        torch.manual_seed(0)
        repr = torch.rand(1, 3, 32, 32)
        guide = torch.rand(1, 1, 32, 32)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

        op = ops.MRFOperator(encoder, patch_size, stride=stride)
        op.register_buffer("target_repr", repr)
        op.set_target_guide(guide, recalc_repr=False)

        actual = op.target_repr
        desired = repr
        self.assertTensorAlmostEqual(actual, desired)

    def test_MRFOperator_call(self):
        patch_size = 3
        stride = 2

        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 32, 32)
        input_image = torch.rand(1, 3, 32, 32)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

        op = ops.MRFOperator(encoder, patch_size, stride=stride)
        op.set_target_image(target_image)

        actual = op(input_image)
        desired = F.mrf_loss(
            pystiche.extract_patches2d(encoder(input_image), patch_size, stride=stride),
            pystiche.extract_patches2d(
                encoder(target_image), patch_size, stride=stride
            ),
        )
        self.assertFloatAlmostEqual(actual, desired)

    def test_MRFOperator_call_guided(self):
        patch_size = 2
        stride = 2

        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 32, 32)
        input_image = torch.rand(1, 3, 32, 32)
        target_guide = torch.cat(
            (torch.zeros(1, 1, 16, 32), torch.ones(1, 1, 16, 32)), dim=2
        )
        input_guide = target_guide.flip(2)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

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

    def test_OperatorContainer_set_image_or_guide(self):
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

        with self.subTest("set_target_guide"):
            container = get_container()
            container.set_target_guide(image_or_guide, recalc_repr=False)

            actual = container.comparison.target_guide
            desired = image_or_guide
            self.assertTensorAlmostEqual(actual, desired)

        with self.subTest("set_target_guide"):
            container = get_container()
            container.set_target_image(image_or_guide)

            actual = container.comparison.target_image
            desired = image_or_guide
            self.assertTensorAlmostEqual(actual, desired)

        with self.subTest("set_input_guide"):
            container = get_container()
            container.set_input_guide(image_or_guide)

            actual = container.regularization.input_guide
            desired = image_or_guide
            self.assertTensorAlmostEqual(actual, desired)

            actual = container.comparison.input_guide
            desired = image_or_guide
            self.assertTensorAlmostEqual(actual, desired)

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

    def test_MultiRegionOperator_set_regional_image_or_guide(self):
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
            with self.subTest(method):
                multi_region_operator = get_multi_region_operator()

                for region, image_or_guide in regional_images_or_guides:
                    getattr(multi_region_operator, method)(region, image_or_guide)

                for region, image_or_guide in regional_images_or_guides:
                    actual = getattr(
                        getattr(multi_region_operator, region), desired_attr
                    )
                    desired = image_or_guide
                    self.assertTensorAlmostEqual(actual, desired)


class TestFunctional(PysticheTestCase):
    def test_mrf_loss(self):
        torch.manual_seed(0)
        zero_patch = torch.zeros(3, 3, 3)
        one_patch = torch.ones(3, 3, 3)
        rand_patch = torch.randn(3, 3, 3)

        input = torch.stack((rand_patch + 0.1, rand_patch * 0.9))
        target = torch.stack((zero_patch, one_patch, rand_patch))

        actual = F.mrf_loss(input, target)
        desired = mse_loss(input, torch.stack((rand_patch, rand_patch)))
        self.assertFloatAlmostEqual(actual, desired)

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


class TestOp(PysticheTestCase):
    def test_Operator_set_input_guide(self):
        class TestOperator(ops.Operator):
            def process_input_image(self, image):
                pass

        torch.manual_seed(0)
        guide = torch.rand(1, 1, 32, 32)

        test_op = TestOperator()
        self.assertFalse(test_op.has_input_guide)

        test_op.set_input_guide(guide)
        self.assertTrue(test_op.has_input_guide)

        actual = test_op.input_guide
        desired = guide
        self.assertTensorAlmostEqual(actual, desired)

    def test_Operator_apply_guide(self):
        torch.manual_seed(0)
        image = torch.rand(1, 3, 32, 32)
        guide = torch.rand(1, 1, 32, 32)

        actual = ops.Operator.apply_guide(image, guide)
        desired = image * guide
        self.assertTensorAlmostEqual(actual, desired)

    def test_Operator_named_operators(self):
        # TODO: add
        pass

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

    def test_PixelRegularizationOperator_call_guided(self):
        class TestOperator(ops.PixelRegularizationOperator):
            def input_image_to_repr(self, image):
                return image * 2.0

            def calculate_score(self, input_repr):
                return input_repr + 1.0

        torch.manual_seed(0)
        image = torch.rand(1, 3, 32, 32)
        guide = torch.rand(1, 1, 32, 32)

        test_op = TestOperator()
        test_op.set_input_guide(guide)

        actual = test_op(image)
        desired = TestOperator.apply_guide(image, guide) * 2.0 + 1.0
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

    def test_EncodingRegularizationOperator_call_guided(self):
        class TestOperator(ops.EncodingRegularizationOperator):
            def input_enc_to_repr(self, image):
                return image * 2.0

            def calculate_score(self, input_repr):
                return input_repr + 1.0

        torch.manual_seed(0)
        image = torch.rand(1, 3, 32, 32)
        guide = torch.rand(1, 3, 32, 32)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))
        enc_guide = encoder.propagate_guide(guide)

        test_op = TestOperator(encoder)
        test_op.set_input_guide(guide)

        actual = test_op(image)
        desired = TestOperator.apply_guide(encoder(image), enc_guide) * 2.0 + 1.0
        self.assertTensorAlmostEqual(actual, desired)

    def test_PixelComparisonOperator_set_target_guide(self):
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
        image = torch.rand(1, 3, 32, 32)
        guide = torch.rand(1, 1, 32, 32)

        test_op = TestOperator()
        test_op.set_target_image(image)
        self.assertFalse(test_op.has_target_guide)

        test_op.set_target_guide(guide)
        self.assertTrue(test_op.has_target_guide)

        actual = test_op.target_guide
        desired = guide
        self.assertTensorAlmostEqual(actual, desired)

        actual = test_op.target_image
        desired = image
        self.assertTensorAlmostEqual(actual, desired)

    def test_PixelComparisonOperator_set_target_guide_without_recalc(self):
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
        repr = torch.rand(1, 3, 32, 32)
        guide = torch.rand(1, 1, 32, 32)

        test_op = TestOperator()
        test_op.register_buffer("target_repr", repr)
        test_op.set_target_guide(guide, recalc_repr=False)

        actual = test_op.target_repr
        desired = repr
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

    def test_PixelComparisonOperator_call_guided(self):
        class TestOperator(ops.PixelComparisonOperator):
            def target_image_to_repr(self, image):
                repr = image + 1.0
                return repr, None

            def input_image_to_repr(self, image, ctx):
                return image + 2.0

            def calculate_score(self, input_repr, target_repr, ctx):
                return input_repr * target_repr

        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 32, 32)
        target_guide = torch.rand(1, 1, 32, 32)
        input_image = torch.rand(1, 3, 32, 32)
        input_guide = torch.rand(1, 1, 32, 32)

        test_op = TestOperator()
        test_op.set_target_guide(target_guide)
        test_op.set_target_image(target_image)
        test_op.set_input_guide(input_guide)

        actual = test_op(input_image)
        desired = (TestOperator.apply_guide(target_image, target_guide) + 1.0) * (
            TestOperator.apply_guide(input_image, input_guide) + 2.0
        )
        self.assertTensorAlmostEqual(actual, desired)

    def test_EncodingComparisonOperator_set_target_guide(self):
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
        image = torch.rand(1, 3, 32, 32)
        guide = torch.rand(1, 1, 32, 32)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))
        enc_guide = encoder.propagate_guide(guide)

        test_op = TestOperator(encoder)
        test_op.set_target_image(image)
        self.assertFalse(test_op.has_target_guide)

        test_op.set_target_guide(guide)
        self.assertTrue(test_op.has_target_guide)

        actual = test_op.target_guide
        desired = guide
        self.assertTensorAlmostEqual(actual, desired)

        actual = test_op.target_enc_guide
        desired = enc_guide
        self.assertTensorAlmostEqual(actual, desired)

        actual = test_op.target_image
        desired = image
        self.assertTensorAlmostEqual(actual, desired)

    def test_EncodingComparisonOperator_set_target_guide_without_recalc(self):
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
        repr = torch.rand(1, 3, 32, 32)
        guide = torch.rand(1, 1, 32, 32)
        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))

        test_op = TestOperator(encoder)
        test_op.register_buffer("target_repr", repr)
        test_op.set_target_guide(guide, recalc_repr=False)

        actual = test_op.target_repr
        desired = repr
        self.assertTensorAlmostEqual(actual, desired)

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

    def test_EncodingComparisonOperator_call_guided(self):
        class TestOperator(ops.EncodingComparisonOperator):
            def target_enc_to_repr(self, image):
                repr = image + 1.0
                return repr, None

            def input_enc_to_repr(self, image, ctx):
                return image + 2.0

            def calculate_score(self, input_repr, target_repr, ctx):
                return input_repr * target_repr

        torch.manual_seed(0)
        target_image = torch.rand(1, 3, 32, 32)
        input_image = torch.rand(1, 3, 32, 32)
        target_guide = torch.rand(1, 1, 32, 32)
        input_guide = torch.rand(1, 1, 32, 32)

        encoder = SequentialEncoder((nn.Conv2d(3, 3, 1),))
        target_enc_guide = encoder.propagate_guide(target_guide)
        input_enc_guide = encoder.propagate_guide(input_guide)

        test_op = TestOperator(encoder)
        test_op.set_target_guide(target_guide)
        test_op.set_target_image(target_image)
        test_op.set_input_guide(input_guide)

        actual = test_op(input_image)
        desired = (
            TestOperator.apply_guide(encoder(target_image), target_enc_guide) + 1.0
        ) * (TestOperator.apply_guide(encoder(input_image), input_enc_guide) + 2.0)
        self.assertTensorAlmostEqual(actual, desired)


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
