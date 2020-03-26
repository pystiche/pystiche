"""
NST via image-based optimisation without pystiche
=================================================
"""

###############################################################################
# We start of the tutorial by importing all necessary packages. Next to torch and
# torchvision, we use PIL and matplotlib.pyplot to read, write, and display images.

import torch

print("I'm working with torch version " + torch.__version__)
from torch import nn
import torch.nn.functional as Fnn
from torch import optim

import torchvision

print("I'm working with torchvision version " + torchvision.__version__)
from torchvision.models import vgg19
from torchvision import transforms
import torchvision.transforms.functional as Fv

from os import path
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt

###############################################################################
# The NST algorithm involves a neural network as part of an optimisation problem. Thus
# it is really helpful to do all calculations on a GPU to speed up the process. This
# tutorial requires only about 2.5 GB of memory.
#
# For this tutorial all operations are executed on a GPU if one is available. If that is
# not the case it will still work correctly but significantly slower.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("I'm working on device: " + str(device))

###############################################################################


class Encoder(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self.eval()

    def forward(self):
        pass


###############################################################################
# Before we can finally dive into the actual NST, two more preliminary steps have to be
# taken care of. The ``torchvision`` package already offers some transformations, but
# we define some additonal ones.
#
# .. note::
#    The functionality of all transformations listed below could be achieved with the
#    ``transforms.Lambda()`` transformation. Unfortunately,
#    ``print(transforms.Lambda())`` would not display any information of what it is
#    doing. Since one of the goals of this tutorial is clarity, this is avoided here.


class Transform(object):
    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.extra_repr())

    def extra_repr(self):
        return ""


class ToCpu(Transform):
    def __call__(self, tensor_image):
        return tensor_image.cpu()


class EnforceFloatPixelValueRange(Transform):
    def __call__(self, tensor_image):
        return torch.clamp(tensor_image, 0.0, 1.0)


class AddFakeBatchDim(Transform):
    def __call__(self, tensor_image):
        return tensor_image.unsqueeze(0)


class RemoveFakeBatchDim(Transform):
    def __call__(self, tensor_image):
        return tensor_image.squeeze(0)


###############################################################################
# To apply all neceessary transformations conveniently, we bundle them together within
# a ``transforms.Compose`` container. We define a ``preprocessor`` that performs the
# following steps:
#
# 1. Given an ``PIL`` image it is cast it into a ``torch.Tensor`` and the dimensions
#    are rearranged to ``CxHxW``.
# 2. A fake batch dimensions is added to be able to pass the image into our encoder.

preprocessor = transforms.Compose((transforms.ToTensor(), AddFakeBatchDim()))
print("I'm working with the following preprocessor:")
print(preprocessor)

###############################################################################
# The ``postprocessor`` is also defined as ``transforms.Compose`` and performs the
# steps of the ``preprocessor`` in reverse as well as two additonal steps:
#
# 1. The image is moved to the CPU before performing any other actions, since the
#    transformations are not defined to work on the GPU.
# 2. Before converting the tensor back to a ``PIL`` image we enforce the float value
#    range for pixels. Since the optimization is unconstrained, it might have created
#    values outside the closed interval :math:`\left[ 0 ,\, 1\right]`.

postprocessor = transforms.Compose(
    (
        ToCpu(),
        RemoveFakeBatchDim(),
        EnforceFloatPixelValueRange(),
        transforms.ToPILImage(),
    )
)
print("I'm working with the following postprocessor:")
print(postprocessor)

###############################################################################
# As a last preliminary step we define some image I/O functions that help us read,
# write, and show images. These helper functions incorporate the above defined
# ``preprocessor`` and ``postprocessor`` so that we don't have to call them explicitly.
#
# :func:`read_image` resizes the input image so that the smallest side is
# ``image_size`` pixels wide while keeping the aspect ratio constant. The default value
# is set to ``image_size=500`` since Gatys et. al. reported in a follow up paper that
#
#   for the VGG-19 network, there is a sweet spot around :math:`500^2` pixels for the
#   size of the input images, such that the stylisation is appealing but the content is
#   well-preserved.


def read_image(file, image_size=500):
    image = Image.open(file)
    image = Fv.resize(image, image_size)
    return preprocessor(image)


def write_image(image, file):
    image = postprocessor(image)
    image.save(file)


def show_image(image, title=None, show_axis=False):
    _, ax = plt.subplots()

    ax.imshow(image)
    if not show_axis:
        ax.axis("off")
    if title is not None:
        ax.set_title(title)


###############################################################################
# Now we put the previously defined encoder to use by creating the target content and
# style encodings for a different set of layers. The layer configuration is taken from
# Gatys et. al.. Since we now know which layers we want to use, unused ones are removed
# from the encoder with the :meth:`~Encoder.trim` method.
