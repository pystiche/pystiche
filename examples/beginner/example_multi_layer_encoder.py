"""
Multi-layer Encoder
===================

This example showcases the :class:`pystiche.enc.MultiLayerEncoder`.

.. warning::

  This example uses features that are only availble for ``pystiche>0.7``, which is not
  released yet. If you want to use these features, head over to the
  :ref:`installation instructions <installation>` and install the
  *potentially unstable* version.
"""


########################################################################################
# We start this example by importing everything we need.

import itertools
import time
from collections import OrderedDict
from math import floor, log10

import torch
from torch import nn
from torchvision import models

import pystiche
from pystiche import enc

print(f"I'm working with pystiche=={pystiche.__version__}")


########################################################################################
# In a second preliminary step we define some helper functions to ease the performance
# analysis later on.

SI_PREFIXES = {0: "", -3: "m", -6: "µ"}


def timeit(fn, times=10, cleanup=None):
    total = 0.0
    for _ in range(times):
        start = time.time()
        fn()
        stop = time.time()
        total += stop - start
        if cleanup:
            cleanup()
    return total / times


def feng(num, unit, digits=3):
    exp = int(floor(log10(num)))
    exp -= exp % 3
    sig = num * 10 ** -exp
    prec = digits - len(str(int(sig)))
    return f"{sig:.{prec}f} {SI_PREFIXES[exp]}{unit}"


def fsecs(seconds):
    return feng(seconds, "s")


def ftimeit(fn, msg="The execution took {seconds}.", **kwargs):
    return msg.format(seconds=fsecs(timeit(fn, **kwargs)))


def fdifftimeit(seq_fn, mle_fn, **kwargs):
    time_seq = timeit(seq_fn, **kwargs)
    time_mle = timeit(mle_fn, **kwargs)

    abs_diff = time_mle - time_seq
    rel_diff = abs_diff / time_seq

    if abs_diff >= 0:
        return (
            f"Encoding the input with the enc.MultiLayerEncoder was "
            f"{fsecs(abs_diff)} ({rel_diff:.0%}) slower."
        )
    else:
        return "\n".join(
            (
                "Due to the very rough timing method used here, ",
                "we detected a case where the encoding with the enc.MultiLayerEncoder ",
                "was actually faster than the boiler-plate nn.Sequential. ",
                "Since the enc.MultiLayerEncoder has some overhead, ",
                "this is a measuring error. ",
                "Still, this serves as indicator that the overhead is small enough, ",
                "to be well in the measuring tolerance.",
            )
        )


########################################################################################
# Next up, we define the device we will be testing on as well as the input dimensions.
#
# .. note::
#
#   We encourage the user to play with these parameters and see how the results change.
#   In order to do that, you can use the download buttons at the bottom of this page.

device = torch.device("cpu")

batch_size = 32
num_channels = 3
height = width = 512

input = torch.rand((batch_size, num_channels, height, width), device=device)


########################################################################################
# As a toy example to showcase the :class:`~pystiche.enc.MultiLayerEncoder`
# capabilities, we will use a CNN with three layers.

conv = nn.Conv2d(num_channels, num_channels, 3, padding=1)
relu = nn.ReLU(inplace=False)
pool = nn.MaxPool2d(2)

modules = [("conv", conv), ("relu", relu), ("pool", pool)]

seq = nn.Sequential(OrderedDict(modules)).to(device)
mle = enc.MultiLayerEncoder(modules).to(device)
print(mle)


########################################################################################
# Before we dive into the additional functionalities of the
# :class:`~pystiche.enc.MultiLayerEncoder` we perform a smoke test and assert that it
# indeed does the same as an :class:`torch.nn.Sequential` with the same layers.

assert torch.allclose(mle(input), seq(input))
print(fdifftimeit(lambda: seq(input), lambda: mle(input)))


########################################################################################
# As we saw, the :class:`~pystiche.enc.MultiLayerEncoder` produces the same output as
# an :class:`torch.nn.Sequential` but is slower. In the following we will learn what
# other functionalities a :class:`~pystiche.enc.MultiLayerEncoder` has to offer that
# justify this overhead.
#
# Intermediate feature maps
# -------------------------
#
# By calling the multi-layer encoder with a layer name in addition to the input, the
# intermediate layers of the :class:`~pystiche.enc.MultiLayerEncoder` can be accessed.
# This is helpful if one needs the feature maps from different layers of a model, as is
# often the case during an NST.

assert torch.allclose(mle(input, "conv"), conv(input))
assert torch.allclose(mle(input, "relu"), relu(conv(input)))
assert torch.allclose(mle(input, "pool"), pool(relu(conv(input))))


########################################################################################
# For convenience, one can extract a :class:`pystiche.enc.SingleLayerEncoder` as an
# interface to the multi-layer encoder for a specific layer.

sle = mle.extract_encoder("conv")
assert torch.allclose(sle(input), conv(input))


########################################################################################
# Caching
# -------
#
# If the access intermediate feature maps is necessary, as is usually the case in an
# NST, it is important to only compute every layer once.
#
# A :meth:`~pystiche.enc.MultiLayerEncoder` enables this functionality by caching
# already computed feature maps. Thus, after an input is cached, retrieving it is a
# constant time lookup
#
# In order to enable caching for a layer, it has to be registered first.
#
# .. note::
#
#   The internal cache has no functionality to clear it automatically. Thus, the user
#   has to manually call :meth:`~pystiche.enc.MultiLayerEncoder.clear_cache` to avoid
#   memory build up. In the builtin optimization functions such as
#   :func:`pystiche.optim.image_optimization` this is performed after every
#   optimization step.
#
# .. note::
#
#   :meth:`~pystiche.enc.MultiLayerEncoder.extract_encoder` automatically registers the
#   layer for caching.

shallow_layers = ("conv", "relu")
for layer in shallow_layers:
    mle.register_layer(layer)

mle(input)

for layer in shallow_layers:
    print(
        ftimeit(
            lambda: mle(input, layer),
            f"The encoding of layer '{layer}' took {{seconds}}.",
        )
    )

mle.clear_cache()


########################################################################################
# Due to this caching, it doesn't matter in which order the feature maps are requested:
#
# 1. If a shallow layer is requested before a deeper one, the encoding is later resumed
#    from the feature map of the shallow layer.
# 2. If a deep layer is requested before a more shallow one, the feature map of the
#    shallow one is cached while computing the deep layer.


def fn(layers):
    for layer in layers:
        mle(input, layer)


for permutation in itertools.permutations(("conv", "relu", "pool")):
    order = f"""'{"' -> '".join(permutation)}'"""
    print(
        ftimeit(
            lambda: fn(permutation),
            f"The encoding of layers {order} took {{seconds}}.",
            cleanup=mle.clear_cache,
        )
    )


########################################################################################
# Real-world example
# ------------------
#
# Up to this point we used a toy example to demonstrate the capabilities of a
# :class:`~pystiche.enc.MultiLayerEncoder`. In addition to the boiler-plate
# :class:`~pystiche.enc.MultiLayerEncoder`, ``pystiche`` has builtin implementations of
# some well-known CNN architectures that are commonly used in NST papers.
#
# .. note::
#
#   By default, :func:`~pystiche.enc.vgg19_multi_layer_encoder` loads weights provided
#   by ``torchvision``. We disable this here since we load the randomly initilaized
#   weights of the ``torchvision`` model to enable a comparison.
#
# .. note::
#
#   By default, :func:`~pystiche.enc.vgg19_multi_layer_encoder` adds an
#   ``internal_preprocessing`` so that the user can simply pass the image as is,
#   without worrying about it. We disable this here to enable a comparison.
#
# .. note::
#
#   By default, :func:`~pystiche.enc.vgg19_multi_layer_encoder` disallows in-place
#   operations since after they are carried out, the previous encoding is no longer
#   accessible. In order to enable a fair performance comparison, we allow them here,
#   since they are also used in :func:`~torchvision.models.vgg19`.
#
# .. note::
#
#   The fully connected stage of the original VGG19 architecture requires the input to
#   be exactly 224 pixels wide and high :cite:`SZ2014`. Since this requirement can
#   usually not be met in an NST, the builtin multi-layer encoder only comprises the
#   size invariant convolutional stage. Thus, we only use ``vgg19().features`` to
#   enable a comparison.

seq = models.vgg19()
mle = enc.vgg19_multi_layer_encoder(
    pretrained=False, internal_preprocessing=False, allow_inplace=True
)
mle.load_state_dict(seq.state_dict())

input = torch.rand((4, 3, 256, 256), device=device)

assert torch.allclose(mle(input), seq.features(input))
print(fdifftimeit(lambda: seq.features(input), lambda: mle(input)))
