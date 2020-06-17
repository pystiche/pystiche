"""
Neural Style Transfer without ``pystiche``
==========================================

This example showcases how a basic Neural Style Transfer (NST), i.e. image-based
optimization, could be performed without ``pystiche``.

.. note::

    This is an *example how to implement an NST* and **not** a
    *tutorial on how NST works*. As such, it will not explain why a specific choice was
    made or how a component works. If you have never worked with NST before, we
    **strongly** suggest you to read the :ref:`gist` first.
"""


########################################################################################
# Setup
# -----
#
# We start this example by importing everything we need and setting the device we will
# be working on. :mod:`torch` and :mod:`torchvision` will be used for the actual NST.
# Furthermore, we use :mod:`requests` for the download, :mod:`PIL.Image` for the file
# input, and :mod:`matplotlib.pyplot` to show the images.

import itertools
from collections import OrderedDict

import matplotlib.pyplot as plt
import requests
from PIL import Image

import torch
import torchvision
from torch import nn, optim
from torch.nn.functional import mse_loss
from torchvision import transforms
from torchvision.models import vgg19
from torchvision.transforms.functional import resize

print(f"I'm working with torch=={torch.__version__}")
print(f"I'm working with torchvision=={torchvision.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"I'm working with {device}")


########################################################################################
# The core component of different NSTs is the perceptual loss, which is used as
# optimization criterion. The perceptual loss is usually, and also for this example,
# calculated on features maps also called encodings. These encodings are generated from
# different layers of a Convolutional Neural Net (CNN) also called encoder.
#
# A common implementation strategy for the perceptual loss is to *weave in* transparent
# loss layers into the encoder. These loss layers are called transparent since from an
# outside view they simply pass the input through without alteration. Internally
# though, they calculate the loss with the encodings of the previous layer and store
# them in themselves. After the forward pass is completed the stored losses are
# aggregated and propagated backwards to the image. While this is simple to implement,
# this practice has two downsides:
#
# 1. The calculated score is part of the current state but has to be stored inside the
#    layer. This is generally not recommended.
# 2. While the encoder is a part of the perceptual loss, it itself does not generate
#    it. One should be able to use the same encoder with a different perceptual loss
#    without modification.
#
# Thus, this example (and ``pystiche``) follows a different approach and separates the
# encoder and the perceptual loss into individual entities.


########################################################################################
# Multi-layer Encoder
# -------------------
#
# In a first step we define a ``MultiLayerEncoder`` that should have the following
# properties:
#
# 1. Given an image and a set of layers, the ``MultiLayerEncoder`` should return the
#    encodings of every given layer.
# 2. Since the encodings have to be generated in every optimization step they should be
#    calculated in a single forward pass to keep the processing costs low.
# 3. To reduce the static memory requirement, the ``MultiLayerEncoder`` should be
#    ``trim`` mable in order to remove unused layers.
#
# We achieve the main functionality by subclassing :class:`torch.nn.Sequential` and
# define a custom ``forward`` method, i.e. different behavior if called. Besides the
# image it also takes an iterable ``layer_cfgs`` containing multiple sequences of
# ``layers``. In the method body we first find the ``deepest_layer`` that was
# requested. Subsequently, we calculate and store all encodings of the ``image`` up to
# that layer. Finally we can return all requested encodings without processing the same
# layer twice.


class MultiLayerEncoder(nn.Sequential):
    def forward(self, image, *layer_cfgs):
        storage = {}
        deepest_layer = self._find_deepest_layer(*layer_cfgs)
        for layer, module in self.named_children():
            image = storage[layer] = module(image)
            if layer == deepest_layer:
                break

        return [[storage[layer] for layer in layers] for layers in layer_cfgs]

    def children_names(self):
        for name, module in self.named_children():
            yield name

    def _find_deepest_layer(self, *layer_cfgs):
        # find all unique requested layers
        req_layers = set(itertools.chain(*layer_cfgs))
        try:
            # find the deepest requested layer by indexing the layers within
            # the multi layer encoder
            children_names = list(self.children_names())
            return sorted(req_layers, key=children_names.index)[-1]
        except ValueError as error:
            layer = str(error).split()[0]
        raise ValueError(f"Layer {layer} is not part of the multi-layer encoder.")

    def trim(self, *layer_cfgs):
        deepest_layer = self._find_deepest_layer(*layer_cfgs)
        children_names = list(self.children_names())
        del self[children_names.index(deepest_layer) + 1 :]


########################################################################################
# The pretrained models the ``MultiLayerEncoder`` is based on are usually trained on
# preprocessed images. In PyTorch all models expect images are
# `normalized <https://pytorch.org/docs/stable/torchvision/models.html>`_ by a
# per-channel ``mean = (0.485, 0.456, 0.406)`` and standard deviation
# (``std = (0.229, 0.224, 0.225)``). To include this into a, ``MultiLayerEncoder``, we
# implement this as :class:`torch.nn.Module` .


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, image):
        return (image - self.mean) / self.std


class TorchNormalize(Normalize):
    def __init__(self):
        super().__init__((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


########################################################################################
# In a last step we need to specify the structure of the ``MultiLayerEncoder``. In this
# example we use a ``VGGMultiLayerEncoder`` based on the ``VGG19`` CNN introduced by
# Simonyan and Zisserman :cite:`SZ2014`.
#
# We only include the feature extraction stage (``vgg_net.features``), i.e. the
# convolutional stage, since the classifier stage (``vgg_net.classifier``) only accepts
# feature maps of a single size.
#
# For our convenience we rename the layers in the same scheme the authors used instead
# of keeping the consecutive index of a default :class:`torch.nn.Sequential`. The first
# layer however is the ``TorchNormalize``  as defined above.


class VGGMultiLayerEncoder(MultiLayerEncoder):
    def __init__(self, vgg_net):
        modules = OrderedDict((("preprocessing", TorchNormalize()),))

        block = depth = 1
        for module in vgg_net.features.children():
            if isinstance(module, nn.Conv2d):
                layer = f"conv{block}_{depth}"
            elif isinstance(module, nn.BatchNorm2d):
                layer = f"bn{block}_{depth}"
            elif isinstance(module, nn.ReLU):
                # without inplace=False the encodings of the previous layer would no
                # longer be accessible after the ReLU layer is executed
                module = nn.ReLU(inplace=False)
                layer = f"relu{block}_{depth}"
                # each ReLU layer increases the depth of the current block by one
                depth += 1
            elif isinstance(module, nn.MaxPool2d):
                layer = f"pool{block}"
                # each max pooling layer marks the end of the current block
                block += 1
                depth = 1
            else:
                msg = f"Type {type(module)} is not part of the VGG architecture."
                raise RuntimeError(msg)

            modules[layer] = module

        super().__init__(modules)


def vgg19_multi_layer_encoder():
    return VGGMultiLayerEncoder(vgg19(pretrained=True))


multi_layer_encoder = vgg19_multi_layer_encoder().to(device)
print(multi_layer_encoder)


########################################################################################
# Perceptual Loss
# ---------------
#
# In order to calculate the perceptual loss, i.e. the optimization criterion, we define
# a ``MultiLayerLoss`` to have a convenient interface. This will be subclassed later by
# the ``ContentLoss`` and ``StyleLoss``.
#
# If called with a sequence of ``ìnput_encs`` the ``MultiLayerLoss`` should calculate
# layerwise scores together with the corresponding ``target_encs``. For that a
# ``MultiLayerLoss`` needs the ability to store the ``target_encs`` so that they can be
# reused for every call. The individual layer scores should be averaged by the number
# of encodings and finally weighted by a ``score_weight``.
#
# To achieve this we subclass :class:`torch.nn.Module` . The ``target_encs`` are stored
# as buffers, since they are not trainable parameters. The actual functionality has to
# be defined in ``calculate_score`` by a subclass.


def mean(sized):
    return sum(sized) / len(sized)


class MultiLayerLoss(nn.Module):
    def __init__(self, score_weight=1e0):
        super().__init__()
        self.score_weight = score_weight
        self._numel_target_encs = 0

    def _target_enc_name(self, idx):
        return f"_target_encs_{idx}"

    def set_target_encs(self, target_encs):
        self._numel_target_encs = len(target_encs)
        for idx, enc in enumerate(target_encs):
            self.register_buffer(self._target_enc_name(idx), enc.detach())

    @property
    def target_encs(self):
        return tuple(
            getattr(self, self._target_enc_name(idx))
            for idx in range(self._numel_target_encs)
        )

    def forward(self, input_encs):
        if len(input_encs) != self._numel_target_encs:
            msg = (
                f"The number of given input encodings and stored target encodings "
                f"does not match: {len(input_encs)} != {self._numel_target_encs}"
            )
            raise RuntimeError(msg)

        layer_losses = [
            self.calculate_score(input, target)
            for input, target in zip(input_encs, self.target_encs)
        ]
        return mean(layer_losses) * self.score_weight

    def calculate_score(self, input, target):
        raise NotImplementedError


########################################################################################
# In this example we use the ``feature_reconstruction_loss`` introduced by Mahendran
# and Vedaldi :cite:`MV2014` as ``ContentLoss`` as well as the ``gram_loss`` introduced
# by Gatys, Ecker, and Bethge :cite:`GEB2016` as ``StyleLoss``.


def feature_reconstruction_loss(input, target):
    return mse_loss(input, target)


class ContentLoss(MultiLayerLoss):
    def calculate_score(self, input, target):
        return feature_reconstruction_loss(input, target)


def channelwise_gram_matrix(x, normalize=True):
    x = torch.flatten(x, 2)
    G = torch.bmm(x, x.transpose(1, 2))
    if normalize:
        return G / x.size()[-1]
    else:
        return G


def gram_loss(input, target):
    return mse_loss(channelwise_gram_matrix(input), channelwise_gram_matrix(target))


class StyleLoss(MultiLayerLoss):
    def calculate_score(self, input, target):
        return gram_loss(input, target)


########################################################################################
# Images
# ------
#
# Before we can load the content and style image, we need to define some basic I/O
# utilities.
#
# At import a fake batch dimension is added to the images to be able to pass it through
# the ``MultiLayerEncoder`` without further modification. This dimension is removed
# again upon export. Furthermore, all images will be resized to ``size=500`` pixels.

import_from_pil = transforms.Compose(
    (
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
        transforms.Lambda(lambda x: x.to(device)),
    )
)

export_to_pil = transforms.Compose(
    (
        transforms.Lambda(lambda x: x.cpu()),
        transforms.Lambda(lambda x: x.squeeze(0)),
        transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
        transforms.ToPILImage(),
    )
)


def download_image(url, file):
    with open(file, "wb") as fh:
        # without User-Agent the access is denied
        headers = {"User-Agent": "pystiche"}
        fh.write(requests.get(url, headers=headers).content)


def read_image(file, size=500):
    image = Image.open(file)
    image = resize(image, size)
    return import_from_pil(image)


def show_image(image, title=None):
    _, ax = plt.subplots()
    ax.axis("off")
    if title is not None:
        ax.set_title(title)

    image = export_to_pil(image)
    ax.imshow(image)


########################################################################################
# With the I/O utilities set up, we now download, read, and show the images that will
# be used in the NST.
#
# .. note::
#
#   The images used in this example are licensed under the permissive
#   `Pixabay License <https://pixabay.com/service/license/>`_ .


########################################################################################

content_url = "https://cdn.pixabay.com/photo/2016/01/14/11/26/bird-1139734_960_720.jpg"
content_file = "bird1.jpg"

download_image(content_url, content_file)
content_image = read_image(content_file)
show_image(content_image, title="Content image")


########################################################################################

style_url = (
    "https://cdn.pixabay.com/photo/2017/07/03/20/17/abstract-2468874_960_720.jpg"
)
style_file = "paint.jpg"

download_image(style_url, style_file)
style_image = read_image(style_file)
show_image(style_image, title="Style image")


########################################################################################
# Neural Style Transfer
# ---------------------
#
# At first we chose the ``content_layers`` and ``style_layers`` on which the encodings
# are compared. With them we ``trim`` the ``multi_layer_encoder`` to remove
# unused layers that otherwise occupy memory.
#
# Afterwards we calculate the target content and style encodings. The calculation is
# performed without a gradient since the gradient of the target encodings is not needed
# for the optimization.

content_layers = ("relu4_2",)
style_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")

multi_layer_encoder.trim(content_layers, style_layers)

with torch.no_grad():
    target_content_encs = multi_layer_encoder(content_image, content_layers)[0]
    target_style_encs = multi_layer_encoder(style_image, style_layers)[0]


########################################################################################
# Next up, we instantiate the ``ContentLoss`` and ``StyleLoss`` with a corresponding
# weight. Afterwards we store the previously calculated target encodings.

content_weight = 1e0
content_loss = ContentLoss(score_weight=content_weight)
content_loss.set_target_encs(target_content_encs)

style_weight = 1e3
style_loss = StyleLoss(score_weight=style_weight)
style_loss.set_target_encs(target_style_encs)


########################################################################################
# We start NST from the ``content_image`` since this way it converges quickly.

input_image = content_image.clone()
show_image(input_image, "Input image")


########################################################################################
# .. note::
#
#   If you want to start from a white noise image instead use
#
#   .. code-block:: python
#
#     input_image = torch.rand_like(content_image)


########################################################################################
# In a last preliminary step we create the optimizer that will be performing the NST.
# Since we want to adapt the pixels of the ``input_image`` directly, we pass it as
# optimization parameters.

optimizer = optim.LBFGS([input_image.requires_grad_(True)], max_iter=1)


########################################################################################
# Finally we run the NST. The loss calculation has to happen inside a ``closure``
# since the :class:`~torch.optim.LBFGS` optimizer could need to
# `reevaluate it multiple times per optimization step <https://pytorch.org/docs/stable/optim.html#optimizer-step-closure>`_
# . This structure is also valid for all other optimizers.

num_steps = 500

for step in range(1, num_steps + 1):

    def closure():
        optimizer.zero_grad()

        input_encs = multi_layer_encoder(input_image, content_layers, style_layers)
        input_content_encs, input_style_encs = input_encs

        content_score = content_loss(input_content_encs)
        style_score = style_loss(input_style_encs)

        perceptual_loss = content_score + style_score
        perceptual_loss.backward()

        if step % 50 == 0:
            print(f"Step {step}")
            print(f"Content loss: {content_score.item():.3e}")
            print(f"Style loss:   {style_score.item():.3e}")
            print("-----------------------")

        return perceptual_loss

    optimizer.step(closure)

output_image = input_image.detach()

########################################################################################
# After the NST we show the resulting image.

# sphinx_gallery_thumbnail_number = 4
show_image(output_image, title="Output image")


########################################################################################
# Conclusion
# ----------
#
# As hopefully has become clear, an NST requires even in its simplest form quite a lot
# of utilities and boilerplate code. This makes it hard to maintain and keep bug free
# as it is easy to lose track of everything.
#
# Judging by the lines of code one could (falsely) conclude that the actual NST is just
# an appendix. If you feel the same you can stop worrying now: in
# :ref:`sphx_glr_galleries_examples_beginner_example_nst_with_pystiche.py` we showcase
# how to achieve the same result with ``pystiche``.
