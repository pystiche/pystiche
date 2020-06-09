"""
Image optimization with pyramid
===============================

This example showcases how an
`image pyramid <https://en.wikipedia.org/wiki/Pyramid_(image_processing)>`_ is
integrated in an NST with ``pystiche``.

With an image pyramid the optimization is not performed on a single but rather on
multiple increasing resolutions. This procedure is often dubbed *coarse-to-fine*, since
on the lower resolutions coarse structures are synthesized whereas on the higher levels
the details are carved out.

This technique has the potential to reduce the convergence time as well as to enhance
the overall result :cite:`LW2016,GEB+2017`.
"""


########################################################################################
# We start this example by importing everything we need and setting the device we will
# be working on.

import time

import pystiche
from pystiche.demo import demo_images, demo_logger
from pystiche.enc import vgg19_multi_layer_encoder
from pystiche.image import show_image
from pystiche.loss import PerceptualLoss
from pystiche.misc import get_device, get_input_image
from pystiche.ops import (
    FeatureReconstructionOperator,
    MRFOperator,
    MultiLayerEncodingOperator,
)
from pystiche.optim import default_image_optim_loop, default_image_pyramid_optim_loop
from pystiche.pyramid import ImagePyramid

print(f"I'm working with pystiche=={pystiche.__version__}")

device = get_device()
print(f"I'm working with {device}")


########################################################################################
# At first we define a :class:`~pystiche.loss.PerceptualLoss` that is used as
# optimization ``criterion``.

multi_layer_encoder = vgg19_multi_layer_encoder()


content_layer = "relu4_2"
content_encoder = multi_layer_encoder.extract_encoder(content_layer)
content_weight = 1e0
content_loss = FeatureReconstructionOperator(
    content_encoder, score_weight=content_weight
)


style_layers = ("relu3_1", "relu4_1")
style_weight = 2e0


def get_style_op(encoder, layer_weight):
    return MRFOperator(encoder, patch_size=3, stride=2, score_weight=layer_weight)


style_loss = MultiLayerEncodingOperator(
    multi_layer_encoder, style_layers, get_style_op, score_weight=style_weight,
)

criterion = PerceptualLoss(content_loss, style_loss).to(device)
print(criterion)


########################################################################################
# Next up, we load and show the images that will be used in the NST.

images = demo_images()
images.download()
size = 500


########################################################################################

content_image = images["bird2"].read(size=size, device=device)
show_image(content_image, title="Content image")
criterion.set_content_image(content_image)


########################################################################################

style_image = images["mosaic"].read(size=size, device=device)
show_image(style_image, title="Style image")
criterion.set_style_image(style_image)


########################################################################################
# Image optimization without pyramid
# ----------------------------------
#
# As a baseline we use a standard image optimization without pyramid.

starting_point = "content"
input_image = get_input_image(starting_point, content_image=content_image)
show_image(input_image, title="Input image")


########################################################################################
# We time the NST performed by :func:`~pystiche.optim.default_image_optim_loop` and
# show the result.

start_without_pyramid = time.time()
output_image = default_image_optim_loop(
    input_image, criterion, num_steps=400, logger=demo_logger()
)
stop_without_pyramid = time.time()

show_image(output_image, title="Output image without pyramid")

########################################################################################

elapsed_time_without_pyramid = stop_without_pyramid - start_without_pyramid
print(
    f"Without pyramid the optimization took {elapsed_time_without_pyramid:.0f} seconds."
)

########################################################################################
# As you can see the small blurry branches on the left side of the image were picked up
# by the style transfer. They distort the mosaic pattern, which minders the quality of
# the result. In the next section we tackle this by focusing on coarse elements first
# and add the details afterwards.


########################################################################################
# Image optimization with pyramid
# -------------------------------
#
# Opposed to the prior examples we now want to perform an NST on multiple resolutions.
# In ``pystiche`` this handled by an :class:`~pystiche.pyramid.ImagePyramid` . The
# resolutions are selected by specifying the ``edge_sizes`` of the images on each level
# . The optimization is performed for ``num_steps`` on the different levels.
#
# The resizing of all images, i.e. ``input_image`` and target images (``content_image``
# and ``style_image``) is handled by the ``pyramid``. For that we need to register the
# perceptual loss (``criterion``) as one of the ``resize_targets``.
#
# .. note::
#
#   By default the ``edge_sizes`` correspond to the shorter ``edge`` of the images. To
#   change that you can pass ``edge="long"``. For fine-grained control you can also
#   pass a sequence comprising ``"short"`` and ``"long"`` to select the ``edge`` for
#   each level separately. Its length has to match the length of ``edge_sizes``.
#
# .. note::
#
#   For a fine-grained control over the number of steps on each level you can pass a
#   sequence to select the ``num_steps`` for each level separately. Its length has to
#   match the length of ``edge_sizes``.

edge_sizes = (250, 500)
num_steps = 200
pyramid = ImagePyramid(edge_sizes, num_steps, resize_targets=(criterion,))
print(pyramid)


########################################################################################
# With a pyramid the NST is performed by
# :func:`~pystiche.optim.default_image_pyramid_optim_loop`. We time the execution and
# show the result afterwards.
#
# .. note::
#
#   We regenerate the ``input_image`` since it was changed inplace during the first
#   optimization.

input_image = get_input_image(starting_point, content_image=content_image)

start_with_pyramid = time.time()
output_image = default_image_pyramid_optim_loop(
    input_image, criterion, pyramid, logger=demo_logger()
)
stop_with_pyramid = time.time()

# sphinx_gallery_thumbnail_number = 5
show_image(output_image, title="Output image with pyramid")


########################################################################################

elapsed_time_with_pyramid = stop_with_pyramid - start_with_pyramid
relative_decrease = 1.0 - elapsed_time_with_pyramid / elapsed_time_without_pyramid
print(
    f"With pyramid the optimization took {elapsed_time_with_pyramid:.0f} seconds. "
    f"This is a {relative_decrease:.0%} decrease."
)


########################################################################################
# With the coarse-to-fine architecture of the image pyramid, the stylization of the
# blurry background branches is reduced leaving the mosaic pattern mostly intact. On
# top of this quality improvement the execution time is significantly lower while
# performing the same number of steps.
