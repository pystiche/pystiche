import pytorch_testing_utils as ptu

import torch
from torch import nn

import pystiche
from pystiche import enc, loss, ops

from tests.asserts import assert_named_modules_identical


def test_MultiOperatorLoss():
    class TestOperator(ops.Operator):
        def process_input_image(self, image):
            pass

    named_ops = [(str(idx), TestOperator()) for idx in range(3)]
    multi_op_loss = loss.MultiOperatorLoss(named_ops)

    actuals = multi_op_loss.named_children()
    desireds = named_ops
    assert_named_modules_identical(actuals, desireds)


def test_MultiOperatorLoss_trim():
    class TestOperator(ops.EncodingOperator):
        def __init__(self, encoder, **kwargs):
            super().__init__(**kwargs)
            self._encoder = encoder

        @property
        def encoder(self):
            return self._encoder

        def forward(self, image):
            pass

    layers = [str(idx) for idx in range(3)]
    modules = [(layer, nn.Module()) for layer in layers]
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    ops_ = (("op", TestOperator(multi_layer_encoder.extract_encoder(layers[0])),),)
    loss.MultiOperatorLoss(ops_, trim=True)

    assert layers[0] in multi_layer_encoder
    assert all(layer not in multi_layer_encoder for layer in layers[1:])


def test_MultiOperatorLoss_call():
    class TestOperator(ops.Operator):
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
    ptu.assert_allclose(actual, desired)


def test_MultiOperatorLoss_call_encode(forward_pass_counter):
    class TestOperator(ops.EncodingOperator):
        def __init__(self, encoder, **kwargs):
            super().__init__(**kwargs)
            self._encoder = encoder

        @property
        def encoder(self):
            return self._encoder

        def forward(self, image):
            return torch.sum(self.encoder(image))

    modules = (("count", forward_pass_counter),)
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    ops_ = [
        (str(idx), TestOperator(multi_layer_encoder.extract_encoder("count")),)
        for idx in range(3)
    ]
    multi_op_loss = loss.MultiOperatorLoss(ops_)

    torch.manual_seed(0)
    input = torch.rand(1, 3, 128, 128)

    multi_op_loss(input)
    actual = forward_pass_counter.count
    desired = 1
    assert actual == desired

    multi_op_loss(input)
    actual = forward_pass_counter.count
    desired = 2
    assert actual == desired
