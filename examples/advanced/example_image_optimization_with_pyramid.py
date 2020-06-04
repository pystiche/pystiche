"""
Image-based optimization with image pyramid
===========================================

This example showcases how an image pyramid is integrated in an NST in ``pystiche`` .

With an image pyramid the optimization is not performed on single but rather on
multiple increasing resolutions. This procedure is often dubbed *coarse-to-fine*, since
on the lower resolutions coarse structures are synthesized whereas on the higher levels
the details are carved out.

This technique has the potential to reduce the convergence time as well as to enhance
the overall result :cite:`LW2016,GEB+2017`.
"""


########################################################################################
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
    MRFOperator,
    MultiLayerEncodingOperator,
)
from pystiche.optim import default_image_pyramid_optim_loop
from pystiche.pyramid import ImagePyramid

print(f"I'm working with pystiche=={pystiche.__version__}")

device = get_device()
print(f"I'm working with {device}")

images = demo_images()
images.download()


########################################################################################
# At first we define a :class:`~pystiche.loss.perceptual.PerceptualLoss` that is used
# as optimization ``criterion``.

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
    patch_size = 3
    return MRFOperator(encoder, patch_size, stride=2, score_weight=layer_weight)


style_loss = MultiLayerEncodingOperator(
    multi_layer_encoder, style_layers, get_style_op, score_weight=style_weight,
)

criterion = PerceptualLoss(content_loss, style_loss).to(device)
print(criterion)


########################################################################################
# Opposed to the prior examples we want to perform an NST on multiple resolutions. In
# ``pystiche`` this handled by an :class:`~pystiche.pyramid.ImagePyramid` . The
# resolutions are selected by specifying the ``edge_sizes`` of the images on each level
# . The optimization is performed for ``num_steps`` on the different levels.
#
# The resizing of all images, i.e. ``input_image`` and target images (``content_image``
# and ``style_image``) is handled by the ``pyramid``. For that we need to register the
# perceptual loss (``criterion``) as ``resize_targets``.
#
# .. note::
#
#   By default the ``edge_sizes`` correspond to the shorter ``edge`` of the images. To
#   change that you can pass ``edge="long"``. For fine-grained control you can also
#   pass a sequence comprising ``"short"`` and ``"long"`` to select the ``edge`` for
#   each level separately.

edge_sizes = (300, 550)
num_steps = 200
pyramid = ImagePyramid(edge_sizes, num_steps, resize_targets=(criterion,))
print(pyramid)


########################################################################################
# Next up, we load and show the images that will be used in the NST.

content_image = images["bird2"].read(device=device)
show_image(content_image, title="Input image")


########################################################################################

style_image = images["mosaic"].read(device=device)
show_image(style_image, title="Output image")


########################################################################################
# Although the images would be automatically resized during the optimization you might
# need to resize them before: if you are working with large source images you might
# run out of memory by setting up the targets of the perceptual loss. In that case it
# is good practice to resize the images upfront to the largest size the ``pyramid``
# will handle.

top_level = pyramid[-1]
content_image = top_level.resize_image(content_image)
style_image = top_level.resize_image(style_image)


########################################################################################
# As a last preliminary step the previously loaded images are set as targets for the
# perceptual loss (``criterion``) and we create the input image.

criterion.set_content_image(content_image)
criterion.set_style_image(style_image)

starting_point = "content"
input_image = get_input_image(starting_point, content_image=content_image)
show_image(input_image, title="Input image")


########################################################################################
# Finally we run the NST with the
# :func:`~pystiche.optim.optim.default_image_pyramid_optim_loop`. If ``get_optimizer``
# is not specified, as is the case here, the
# :func:`~pystiche.optim.optim.default_image_optimizer`, i.e.
# :class:`~torch.optim.lbfgs.LBFGS` is used.

output_image = default_image_pyramid_optim_loop(
    input_image, criterion, pyramid, logger=demo_logger()
)

# sphinx_gallery_thumbnail_number = 4
show_image(output_image, title="Output image")
