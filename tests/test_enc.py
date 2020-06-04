from os import path

import pytest

import torch
from torch import nn

import pystiche
from pystiche import enc
from pystiche.image.transforms import CaffePreprocessing, TorchPreprocessing

from .utils import ForwardPassCounter, PysticheTestCase


class TestEncoder(PysticheTestCase):
    def test_SequentialEncoder_call(self):
        torch.manual_seed(0)
        modules = (nn.Conv2d(3, 3, 3), nn.ReLU())
        input = torch.rand(1, 3, 256, 256)

        pystiche_encoder = enc.SequentialEncoder(modules)
        torch_encoder = nn.Sequential(*modules)

        actual = pystiche_encoder(input)
        desired = torch_encoder(input)
        self.assertTensorAlmostEqual(actual, desired)


class TestModels(PysticheTestCase):
    @pytest.mark.large_download
    @pytest.mark.slow
    @pytest.mark.flaky
    def test_AlexNetMultiLayerEncoder(self):
        asset = self.load_asset(path.join("enc", "alexnet"))

        multi_layer_encoder = enc.alexnet_multi_layer_encoder(
            weights="torch", preprocessing=False, allow_inplace=False
        )
        layers = tuple(multi_layer_encoder.children_names())
        with torch.no_grad():
            encs = multi_layer_encoder(asset.input.image, layers)

        actual = dict(
            zip(
                layers,
                [pystiche.TensorKey(x, precision=asset.params.precision) for x in encs],
            )
        )
        desired = asset.output.enc_keys
        self.assertDictEqual(actual, desired)

    @pytest.mark.large_download
    @pytest.mark.slow
    def test_alexnet_multi_layer_encoder_smoke(self):
        multi_layer_encoder = enc.alexnet_multi_layer_encoder()
        self.assertIsInstance(multi_layer_encoder, enc.alexnet.AlexNetMultiLayerEncoder)

    @pytest.mark.large_download
    @pytest.mark.slow
    @pytest.mark.flaky
    def test_VGGMultiLayerEncoder(self):
        archs = ("vgg11", "vgg13", "vgg16", "vgg19")
        archs = (*archs, *[f"{arch}_bn" for arch in archs])

        for arch in archs:
            with self.subTest(arch=arch):
                asset = self.load_asset(path.join("enc", arch))

                get_vgg_multi_layer_encoder = enc.__dict__[
                    f"{arch}_multi_layer_encoder"
                ]
                multi_layer_encoder = get_vgg_multi_layer_encoder(
                    weights="torch", preprocessing=False, allow_inplace=False
                )
                layers = tuple(multi_layer_encoder.children_names())
                with torch.no_grad():
                    encs = multi_layer_encoder(asset.input.image, layers)

                actual = dict(
                    zip(
                        layers,
                        [
                            pystiche.TensorKey(x, precision=asset.params.precision)
                            for x in encs
                        ],
                    )
                )
                desired = asset.output.enc_keys
                self.assertDictEqual(actual, desired)

    @pytest.mark.large_download
    @pytest.mark.slow
    def test_vgg_multi_layer_encoder_smoke(self):
        fns = (
            enc.vgg11_multi_layer_encoder,
            enc.vgg11_bn_multi_layer_encoder,
            enc.vgg13_multi_layer_encoder,
            enc.vgg13_bn_multi_layer_encoder,
            enc.vgg16_multi_layer_encoder,
            enc.vgg16_bn_multi_layer_encoder,
            enc.vgg19_multi_layer_encoder,
            enc.vgg19_bn_multi_layer_encoder,
        )
        for fn in fns:
            with self.subTest(fn=fn.__name__):
                multi_layer_encoder = fn()
                self.assertIsInstance(multi_layer_encoder, enc.vgg.VGGMultiLayerEncoder)


class TestMultiLayerEncoder(PysticheTestCase):
    def test_MultiLayerEncoder(self):
        modules = [(str(idx), nn.Module()) for idx in range(3)]
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        for name, module in modules:
            actual = getattr(multi_layer_encoder, name)
            desired = module
            self.assertIs(actual, desired)

    def test_MultiLayerEncoder_named_children(self):
        modules = [(str(idx), nn.Module()) for idx in range(3)]
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        actual = tuple(multi_layer_encoder.children_names())
        desired = tuple(zip(*modules))[0]
        self.assertTupleEqual(actual, desired)

    def test_MultiLayerEncoder_contains(self):
        idcs = (0, 2)
        modules = [(str(idx), nn.Module()) for idx in idcs]
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        for idx in idcs:
            self.assertTrue(str(idx) in multi_layer_encoder)

        for idx in set(range(max(idcs))) - set(idcs):
            self.assertFalse(str(idx) in multi_layer_encoder)

    def test_MultiLayerEncoder_extract_deepest_layer(self):
        layers = [str(idx) for idx in range(3)]
        modules = [(layer, nn.Module()) for layer in layers]
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        actual = multi_layer_encoder.extract_deepest_layer(layers)
        desired = layers[-1]
        self.assertEqual(actual, desired)

        actual = multi_layer_encoder.extract_deepest_layer(sorted(layers, reverse=True))
        desired = layers[-1]
        self.assertEqual(actual, desired)

        del multi_layer_encoder._modules[layers[-1]]

        with self.assertRaises(ValueError):
            multi_layer_encoder.extract_deepest_layer(layers)

        layers = layers[:-1]

        actual = multi_layer_encoder.extract_deepest_layer(layers)
        desired = layers[-1]
        self.assertEqual(actual, desired)

    def test_MultiLayerEncoder_named_children_to(self):
        layers = [str(idx) for idx in range(3)]
        modules = [(layer, nn.Module()) for layer in layers]
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        actuals = multi_layer_encoder.named_children_to(layers[-2])
        desireds = modules[:-2]
        self.assertNamedChildrenEqual(actuals, desireds)

        actuals = multi_layer_encoder.named_children_to(layers[-2], include_last=True)
        desireds = modules[:-1]
        self.assertNamedChildrenEqual(actuals, desireds)

    def test_MultiLayerEncoder_named_children_from(self):
        layers = [str(idx) for idx in range(3)]
        modules = [(layer, nn.Module()) for layer in layers]
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        actuals = multi_layer_encoder.named_children_from(layers[-2])
        desireds = modules[1:]
        self.assertNamedChildrenEqual(actuals, desireds)

        actuals = multi_layer_encoder.named_children_from(
            layers[-2], include_first=False
        )
        desireds = modules[2:]
        self.assertNamedChildrenEqual(actuals, desireds)

    def test_MultiLayerEncoder_call(self):
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 1, 1)
        relu = nn.ReLU(inplace=False)
        pool = nn.MaxPool2d(2)
        input = torch.rand(1, 3, 128, 128)

        modules = (("conv", conv), ("relu", relu), ("pool", pool))
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        layers = ("conv", "pool")
        encs = multi_layer_encoder(input, layers)

        actual = encs[0]
        desired = conv(input)
        self.assertTensorAlmostEqual(actual, desired)

        actual = encs[1]
        desired = pool(relu(conv(input)))
        self.assertTensorAlmostEqual(actual, desired)

    def test_MultiLayerEncoder_call_store(self):
        torch.manual_seed(0)
        count = ForwardPassCounter()

        modules = (("count", count),)
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        layers = ("count",)
        input = torch.rand(1, 3, 128, 128)
        multi_layer_encoder(input, layers, store=True)
        multi_layer_encoder(input, layers)

        actual = count.count
        desired = 1
        self.assertEqual(actual, desired)

        new_input = torch.rand(1, 3, 128, 128)
        multi_layer_encoder(new_input, layers)

        actual = count.count
        desired = 2
        self.assertEqual(actual, desired)

    def test_MultiLayerEncoder_extract_encoder(self):
        conv = nn.Conv2d(3, 1, 1)
        relu = nn.ReLU(inplace=False)

        modules = (("conv", conv), ("relu", relu))
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        layer = "relu"
        single_layer_encoder = multi_layer_encoder.extract_encoder(layer)

        self.assertIsInstance(single_layer_encoder, enc.SingleLayerEncoder)
        self.assertIs(single_layer_encoder.multi_layer_encoder, multi_layer_encoder)
        self.assertEqual(single_layer_encoder.layer, layer)

        self.assertTrue(layer in multi_layer_encoder.registered_layers)

    def test_MultiLayerEncoder_encode(self):
        torch.manual_seed(0)
        count = ForwardPassCounter()
        conv = nn.Conv2d(3, 1, 1)
        relu = nn.ReLU(inplace=False)
        input = torch.rand(1, 3, 128, 128)

        modules = (("count", count), ("conv", conv), ("relu", relu))
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        layers = ("conv", "relu")
        multi_layer_encoder.registered_layers.update(layers)
        multi_layer_encoder.encode(input)
        encs = multi_layer_encoder(input, layers)

        actual = encs[0]
        desired = conv(input)
        self.assertTensorAlmostEqual(actual, desired)

        actual = encs[1]
        desired = relu(conv(input))
        self.assertTensorAlmostEqual(actual, desired)

        actual = count.count
        desired = 1
        self.assertEqual(actual, desired)

    def test_MultiLayerEncoder_empty_storage(self):
        torch.manual_seed(0)
        count = ForwardPassCounter()
        input = torch.rand(1, 3, 128, 128)

        modules = (("count", count),)
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        layers = ("count",)
        multi_layer_encoder(input, layers, store=True)
        multi_layer_encoder.empty_storage()
        multi_layer_encoder(input, layers)

        actual = count.count
        desired = 2
        self.assertEqual(actual, desired)

    def test_MultiLayerEncoder_trim(self):
        layers = [str(idx) for idx in range(3)]
        modules = [(layer, nn.Module()) for layer in layers]
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        for name, module in modules:
            actual = getattr(multi_layer_encoder, name)
            desired = module
            self.assertIs(actual, desired)

        idx = 1
        multi_layer_encoder.trim((str(idx),))

        for name, module in modules[: idx + 1]:
            actual = getattr(multi_layer_encoder, name)
            desired = module
            self.assertIs(actual, desired)

        for name in tuple(zip(*modules))[0][idx + 1 :]:
            with self.assertRaises(AttributeError):
                getattr(multi_layer_encoder, name)

    def test_MultiLayerEncoder_trim_layers(self):
        layers = [str(idx) for idx in range(3)]
        modules = [(layer, nn.Module()) for layer in layers]
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        for name, module in modules:
            actual = getattr(multi_layer_encoder, name)
            desired = module
            self.assertIs(actual, desired)

        idx = 1
        multi_layer_encoder.registered_layers.update(
            [str(idx) for idx in range(idx + 1)]
        )
        multi_layer_encoder.trim()

        for name, module in modules[: idx + 1]:
            actual = getattr(multi_layer_encoder, name)
            desired = module
            self.assertIs(actual, desired)

        for name in tuple(zip(*modules))[0][idx + 1 :]:
            with self.assertRaises(AttributeError):
                getattr(multi_layer_encoder, name)

    def test_SingleLayerEncoder_call(self):
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 1, 1)
        relu = nn.ReLU(inplace=False)
        input = torch.rand(1, 3, 128, 128)

        modules = (("conv", conv), ("relu", relu))
        multi_layer_encoder = enc.MultiLayerEncoder(modules)

        single_layer_encoder = enc.SingleLayerEncoder(multi_layer_encoder, "conv")

        actual = single_layer_encoder(input)
        desired = conv(input)
        self.assertTensorAlmostEqual(actual, desired)


class TestProcessing(PysticheTestCase):
    def test_get_preprocessor(self):
        get_preprocessor = enc.preprocessing.get_preprocessor
        self.assertIsInstance(get_preprocessor("torch"), TorchPreprocessing)
        self.assertIsInstance(get_preprocessor("caffe"), CaffePreprocessing)
