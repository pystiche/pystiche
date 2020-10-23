import pytest
import pytorch_testing_utils as ptu

import torch
from torch import nn

from pystiche import enc, ops


def test_Operator_set_input_guide():
    class TestOperator(ops.Operator):
        def process_input_image(self, image):
            pass

    torch.manual_seed(0)
    guide = torch.rand(1, 1, 32, 32)

    test_op = TestOperator()
    assert not test_op.has_input_guide

    test_op.set_input_guide(guide)
    assert test_op.has_input_guide

    actual = test_op.input_guide
    desired = guide
    ptu.assert_allclose(actual, desired)


def test_Operator_apply_guide():
    torch.manual_seed(0)
    image = torch.rand(1, 3, 32, 32)
    guide = torch.rand(1, 1, 32, 32)

    actual = ops.Operator.apply_guide(image, guide)
    desired = image * guide
    ptu.assert_allclose(actual, desired)


def test_Operator_named_operators():
    # TODO: add
    pass


def test_Operator_call():
    class TestOperator(ops.Operator):
        def process_input_image(self, image):
            return image + 1.0

    torch.manual_seed(0)
    image = torch.rand(1, 3, 128, 128)

    test_op = TestOperator(score_weight=2.0)

    actual = test_op(image)
    desired = (image + 1.0) * 2.0
    ptu.assert_allclose(actual, desired)


def test_PixelRegularizationOperator_call():
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
    ptu.assert_allclose(actual, desired)


def test_PixelRegularizationOperator_call_guided():
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
    ptu.assert_allclose(actual, desired)


def test_EncodingRegularizationOperator_call():
    class TestOperator(ops.EncodingRegularizationOperator):
        def input_enc_to_repr(self, image):
            return image * 2.0

        def calculate_score(self, input_repr):
            return input_repr + 1.0

    torch.manual_seed(0)
    image = torch.rand(1, 3, 128, 128)
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    test_op = TestOperator(encoder)

    actual = test_op(image)
    desired = encoder(image) * 2.0 + 1.0
    ptu.assert_allclose(actual, desired)


def test_EncodingRegularizationOperator_call_guided():
    class TestOperator(ops.EncodingRegularizationOperator):
        def input_enc_to_repr(self, image):
            return image * 2.0

        def calculate_score(self, input_repr):
            return input_repr + 1.0

    torch.manual_seed(0)
    image = torch.rand(1, 3, 32, 32)
    guide = torch.rand(1, 3, 32, 32)
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))
    enc_guide = encoder.propagate_guide(guide)

    test_op = TestOperator(encoder)
    test_op.set_input_guide(guide)

    actual = test_op(image)
    desired = TestOperator.apply_guide(encoder(image), enc_guide) * 2.0 + 1.0
    ptu.assert_allclose(actual, desired)


def test_PixelComparisonOperator_set_target_guide():
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
    assert not test_op.has_target_guide

    test_op.set_target_guide(guide)
    assert test_op.has_target_guide

    actual = test_op.target_guide
    desired = guide
    ptu.assert_allclose(actual, desired)

    actual = test_op.target_image
    desired = image
    ptu.assert_allclose(actual, desired)


def test_PixelComparisonOperator_set_target_guide_without_recalc():
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
    ptu.assert_allclose(actual, desired)


def test_PixelComparisonOperator_set_target_image():
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
    assert not test_op.has_target_image

    test_op.set_target_image(image)
    assert test_op.has_target_image

    actual = test_op.target_image
    desired = image
    ptu.assert_allclose(actual, desired)

    actual = test_op.target_repr
    desired = image * 2.0
    ptu.assert_allclose(actual, desired)

    actual = test_op.ctx
    desired = torch.norm(image)
    ptu.assert_allclose(actual, desired)


def test_PixelComparisonOperator_call():
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
    ptu.assert_allclose(actual, desired)


def test_PixelComparisonOperator_call_no_target():
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

    with pytest.raises(RuntimeError):
        test_op(input_image)


def test_PixelComparisonOperator_call_batch_size_mismatch():
    class TestOperator(ops.PixelComparisonOperator):
        def __init__(self):
            super().__init__()
            self.batch_size_equal = False

        def target_image_to_repr(self, image):
            return image, None

        def input_image_to_repr(self, image, ctx):
            return image

        def calculate_score(self, input_repr, target_repr, ctx):
            input_batch_size = input_repr.size()[0]
            target_batch_size = target_repr.size()[0]
            self.batch_size_equal = input_batch_size == target_batch_size
            return 0.0

    torch.manual_seed(0)
    target_image = torch.rand(1, 3, 128, 128)
    input_image = torch.rand(2, 3, 128, 128)

    test_op = TestOperator()
    test_op.set_target_image(target_image)

    test_op(input_image)
    assert test_op.batch_size_equal


def test_PixelComparisonOperator_call_batch_size_error():
    class TestOperator(ops.PixelComparisonOperator):
        def __init__(self):
            super().__init__()
            self.batch_size_equal = False

        def target_image_to_repr(self, image):
            return image, None

        def input_image_to_repr(self, image, ctx):
            return image

        def calculate_score(self, input_repr, target_repr, ctx):
            pass

    torch.manual_seed(0)
    target_image = torch.rand(2, 1, 1, 1)
    input_image = torch.rand(1, 1, 1, 1)

    test_op = TestOperator()
    test_op.set_target_image(target_image)

    with pytest.raises(RuntimeError):
        test_op(input_image)


def test_PixelComparisonOperator_call_guided():
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
    ptu.assert_allclose(actual, desired)


def test_EncodingComparisonOperator_set_target_guide():
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
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))
    enc_guide = encoder.propagate_guide(guide)

    test_op = TestOperator(encoder)
    test_op.set_target_image(image)
    assert not test_op.has_target_guide

    test_op.set_target_guide(guide)
    assert test_op.has_target_guide

    actual = test_op.target_guide
    desired = guide
    ptu.assert_allclose(actual, desired)

    actual = test_op.target_enc_guide
    desired = enc_guide
    ptu.assert_allclose(actual, desired)

    actual = test_op.target_image
    desired = image
    ptu.assert_allclose(actual, desired)


def test_EncodingComparisonOperator_set_target_guide_without_recalc():
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
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    test_op = TestOperator(encoder)
    test_op.register_buffer("target_repr", repr)
    test_op.set_target_guide(guide, recalc_repr=False)

    actual = test_op.target_repr
    desired = repr
    ptu.assert_allclose(actual, desired)


def test_EncodingComparisonOperator_set_target_image():
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
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    test_op = TestOperator(encoder)
    assert not test_op.has_target_image

    test_op.set_target_image(image)
    assert test_op.has_target_image

    actual = test_op.target_image
    desired = image
    ptu.assert_allclose(actual, desired)

    actual = test_op.target_repr
    desired = encoder(image) * 2.0
    ptu.assert_allclose(actual, desired)

    actual = test_op.ctx
    desired = torch.norm(encoder(image))
    ptu.assert_allclose(actual, desired)


def test_EncodingComparisonOperator_call():
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
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    test_op = TestOperator(encoder)
    test_op.set_target_image(target_image)

    actual = test_op(input_image)
    desired = (encoder(target_image) + 1.0) * (encoder(input_image) + 2.0)
    ptu.assert_allclose(actual, desired)


def test_EncodingComparisonOperator_call_no_target():
    class TestOperator(ops.EncodingComparisonOperator):
        def target_enc_to_repr(self, image):
            pass

        def input_enc_to_repr(self, image, ctx):
            pass

        def calculate_score(self, input_repr, target_repr, ctx):
            pass

    torch.manual_seed(0)
    input_image = torch.rand(1, 3, 128, 128)
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    test_op = TestOperator(encoder)

    with pytest.raises(RuntimeError):
        test_op(input_image)


def test_EncodingComparisonOperator_call_batch_size_mismatch():
    class TestOperator(ops.EncodingComparisonOperator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_size_equal = False

        def target_enc_to_repr(self, enc):
            return enc, None

        def input_enc_to_repr(self, enc, ctx):
            return enc

        def calculate_score(self, input_repr, target_repr, ctx):
            input_batch_size = input_repr.size()[0]
            target_batch_size = target_repr.size()[0]
            self.batch_size_equal = input_batch_size == target_batch_size
            return 0.0

    torch.manual_seed(0)
    target_image = torch.rand(1, 1, 1, 1)
    input_image = torch.rand(2, 1, 1, 1)
    encoder = enc.SequentialEncoder((nn.Conv2d(1, 1, 1),))

    test_op = TestOperator(encoder)
    test_op.set_target_image(target_image)

    test_op(input_image)
    assert test_op.batch_size_equal


def test_EncodingComparisonOperator_call_batch_size_error():
    class TestOperator(ops.EncodingComparisonOperator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_size_equal = False

        def target_enc_to_repr(self, enc):
            return enc, None

        def input_enc_to_repr(self, enc, ctx):
            return enc

        def calculate_score(self, input_repr, target_repr, ctx):
            pass

    torch.manual_seed(0)
    target_image = torch.rand(2, 1, 1, 1)
    input_image = torch.rand(1, 1, 1, 1)
    encoder = enc.SequentialEncoder((nn.Conv2d(1, 1, 1),))

    test_op = TestOperator(encoder)
    test_op.set_target_image(target_image)

    with pytest.raises(RuntimeError):
        test_op(input_image)


def test_EncodingComparisonOperator_call_guided():
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

    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))
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
    ptu.assert_allclose(actual, desired)
