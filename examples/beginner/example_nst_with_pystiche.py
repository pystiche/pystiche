"""
Neural Style Transfer with ``pystiche``
=======================================

This example showcases how a basic Neural Style Transfer (NST), i.e. image-based
optimization, could be performed with ``pystiche``.

.. note::

    This is an *example how to implement an NST* and **not** a
    *tutorial on how NST works*. As such, it will not explain why a specific choice was
    made or how a component works.
"""

########################################################################################
# Setup
# -----
#
# We start this example by importing everything we need and setting the device we will
# be working on.

import pystiche
from pystiche.demo import demo_images, demo_logger
from pystiche.enc import vgg19_multi_layer_encoder
from pystiche.image import show_image
from pystiche.loss import PerceptualLoss
from pystiche.misc import get_device, get_input_image
from pystiche.ops import (
    FeatureReconstructionOperator,
    GramOperator,
    MultiLayerEncodingOperator,
)
from pystiche.optim import default_image_optim_loop

print(f"I'm working with pystiche=={pystiche.__version__}")

device = get_device()
print(f"I'm working with {device}")

images = demo_images()
images.download()


########################################################################################
# .. note::
#
#   ``ìmages.download()`` downloads **all** demo images upfront. If you only want to
#   download the images for this example remove this line. They will be downloaded at
#   runtime instead.


########################################################################################
# Multi-layer Encoder
# -------------------
# The ``content_loss`` and the ``style_loss`` operate on the encodings of an image
# rather than on the image itself. These encodings are generated by a pretrained model
# called encoder. Since we will be using encodings from multiple layers we load a
# multi-layer encoder. In this example we use the ``vgg19_multi_layer_encoder`` that is
# based on the ``VGG19`` architecture introduced by Simonyan and Zisserman
# :cite:`SZ2014` .

multi_layer_encoder = vgg19_multi_layer_encoder()
print(multi_layer_encoder)


########################################################################################
# Perceptual Loss
# ---------------
#
# The core components of every NST are the ``content_loss`` and the ``style_loss``.
# Combined they make up the perceptual loss, i.e. the optimization criterion. In this
# example we use the ``feature_reconstruction_loss`` introduced by Mahendran and
# Vedaldi :cite:`MV2014` as ``content_loss``.
#
# For that we first extract the ``content_encoder`` that generates encodings from the
# ``content_layer``. Together with the ``content_weight`` we initialize a
# :class:`~pystiche.ops.comparison.FeatureReconstructionOperator` serving as content
# loss.

content_layer = "relu4_2"
content_encoder = multi_layer_encoder.extract_single_layer_encoder(content_layer)
content_weight = 1e0
content_loss = FeatureReconstructionOperator(
    content_encoder, score_weight=content_weight
)
print(content_loss)


########################################################################################
# We use the ``gram_loss`` introduced by Gatys, Ecker, and Bethge :cite:`GEB2016` as
# ``style_loss``. Other than before we use multiple ``style_layers``. The individual
# :class:`~pystiche.ops.comparison.GramOperator` s can be conveniently bundled in a
# :class:`~pystiche.ops.container.MultiLayerEncodingOperator`.

style_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
style_weight = 1e3


def get_style_op(encoder, layer_weight):
    return GramOperator(encoder, score_weight=layer_weight)


style_loss = MultiLayerEncodingOperator(
    multi_layer_encoder, style_layers, get_style_op, score_weight=style_weight,
)
print(style_loss)


########################################################################################
# We combine the ``content_loss`` and ``style_loss`` into a joined
# :class:`~pystiche.loss.perceptual.PerceptualLoss`, which will serve as ``criterion``
# for the optimization.

criterion = PerceptualLoss(content_loss, style_loss).to(device)
print(criterion)


########################################################################################
# Images
# ------
#
# We now load and show the images that will be used in the NST.

size = 500


########################################################################################
#
# .. note::
#
#   By default all images will be resized to ``size=500`` pixels on the shorter edge.
#   If you have more memory than X.X GB available you can increase this to obtain
#   higher resolution results.
#
# .. note::
#
#   If you want to work with other images you can load them with
#   :func:`~pystiche.image.io.read_image`:
#
#   .. code-block:: python
#
#     from pystiche.image import read_image
#
#     my_image = read_image("my_image.jpg", size=size, device=device)


########################################################################################

content_image = images["bird1"].read(size=size, device=device)
show_image(content_image, title="Content image")


########################################################################################

style_image = images["paint"].read(size=size, device=device)
show_image(style_image, title="Style image")


########################################################################################
# Neural Style Transfer
# ---------------------
#
# After loading the images they need to be set as targets for the optimization
# ``criterion``.

criterion.set_content_image(content_image)
criterion.set_style_image(style_image)


###############################################################################
# As a last preliminary step we create the input image. We start from the
# ``content_image`` since this way the NST converges quickly.
#
# .. note::
#
#   If you want to start from a white noise image instead use
#   ``starting_point = "random"`` instead:
#
#   .. code-block:: python
#
#     starting_point = "random"
#     input_image = get_input_image(starting_point, content_image=content_image)

starting_point = "content"
input_image = get_input_image(starting_point, content_image=content_image)
show_image(input_image, title="Input image")


########################################################################################
# Finally we run the NST with the
# :func:`~pystiche.optim.optim.default_image_optim_loop`.
# The optimization runs on each ``level`` for ``level.num_steps``.
#
#
# In every step perceptual loss is calculated
# with the ``criterion`` and propagated backward to the ``input_image``. If
# ``get_optimizer`` is not specified, as is the case here, the
# :func:`~pystiche.optim.optim.default_image_optimizer`, i.e.
# :class:`~torch.optim.lbfgs.LBFGS` is used.
#
# .. note::
#
#   By default ``pystiche`` logs the time during an optimization. In order to reduce
#   the clutter, we use the minimal ``demo_logger`` here.

output_image = default_image_optim_loop(
    input_image, criterion, num_steps=500, logger=demo_logger()
)


########################################################################################
# After the NST is complete we show the result.

# sphinx_gallery_thumbnail_number = 4
show_image(output_image, title="Output image")


########################################################################################
# Conclusion
# ----------
#
# If you started with the basic NST example without ``pystiche`` this example hopefully
# convinced you that ``pystiche`` is helpful tool. But this was just the beginning: to
# unleash its full potential head over to the more advanced examples.
