"""
NST via image-based optimization with image pyramid
===================================================
"""


###############################################################################
# imports

import torch
from torch import optim

from pystiche.demo import demo_images
from pystiche.enc import vgg19_multi_layer_encoder
from pystiche.image import extract_aspect_ratio, show_image, write_image
from pystiche.loss import PerceptualLoss
from pystiche.ops import GramOperator, MSEEncodingOperator, MultiLayerEncodingOperator
from pystiche.pyramid import ImagePyramid

###############################################################################
# Make this demo device-agnostic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# Load the encoder used to create the feature maps for the NST

multi_layer_encoder = vgg19_multi_layer_encoder()


###############################################################################
# Create the content loss

content_layer = "relu4_2"
content_encoder = multi_layer_encoder.extract_single_layer_encoder(content_layer)
content_weight = 1e0
content_loss = MSEEncodingOperator(content_encoder, score_weight=content_weight)


###############################################################################
# Create the style loss

style_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
style_weight = 1e4


def get_style_op(encoder, layer_weight):
    return GramOperator(encoder, score_weight=layer_weight)


style_loss = MultiLayerEncodingOperator(
    multi_layer_encoder, style_layers, get_style_op, score_weight=style_weight,
)


###############################################################################
# Combine the content and style loss into the optimization criterion

criterion = PerceptualLoss(content_loss, style_loss).to(device)
print(criterion)


###############################################################################
# Create the image pyramid used for the stylization

edge_sizes = (500, 700)
num_steps = (500, 200)
pyramid = ImagePyramid(edge_sizes, num_steps, resize_targets=(criterion,))


###############################################################################
# load the content and style images and transfer them to the selected device

images = demo_images()
content_image = images["dancing"].read(device=device)
style_image = images["picasso"].read(device=device)


###############################################################################
# resize the images, since the stylization is memory intensive

resize = pyramid[-1].resize_image
content_image = resize(content_image)
style_image = resize(style_image)
show_image(content_image)
show_image(style_image)


###############################################################################
# Set the target images for the content and style loss

criterion.set_content_image(content_image)
criterion.set_style_image(style_image)


###############################################################################
# Set the starting point of the stylization to the content image. If you want
# to start from a white noise image instead, uncomment the line below

input_image = content_image.clone()


###############################################################################
# .. note::
#   To avoid boilerplate code, you can achieve the same behavior with
#   :func:`~pystiche.misc.misc.get_input_image`::
#
#     from pystiche.misc import get_input_image
#
#     starting_point = "content"
#     input_image = get_input_image(starting_point, content_image=content_image)
#
# .. note::
#   If you want to start the stylization from a white noise image instead, you
#   can use::
#
#     input_image = torch.rand_like(content_image)
#
#   or::
#
#     starting_point = "random"
#     input_image = get_input_image(starting_point, content_image=content_image)


###############################################################################
# extract the original aspect ratio to avoid size mismatch errors during resizing

aspect_ratio = extract_aspect_ratio(input_image)


###############################################################################
# Define a getter for the optimizer that performs the stylization


def get_optimizer(input_image):
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


###############################################################################
# Run the stylization

for num_level, level in enumerate(pyramid, 1):
    input_image = level.resize_image(input_image, aspect_ratio=aspect_ratio)
    optimizer = get_optimizer(input_image)

    for step in level:

        def closure():
            optimizer.zero_grad()
            loss = criterion(input_image)
            loss.backward()

            if step % 50 == 0:
                print(f"Level {num_level}, Step {step}")
                print()
                print(loss.aggregate(1))
                print("-" * 80)

            return loss

        optimizer.step(closure)


###############################################################################
# .. note::
#   To avoid boilerplate code, you can achieve the same behavior with
#   :func:`~pystiche.optim.optim.default_image_pyramid_optim_loop`::
#
#     from pystiche.optim import default_image_pyramid_optim_loop
#
#     input_image = default_image_pyramid_optim_loop(
#         input_image, criterion, pyramid, get_optimizer=get_optimizer
#     )
#
#   If you do not pass ``get_optimizer``
#   :func:`~pystiche.optim.optim.default_image_optimizer` is used.


###############################################################################
# Show the stylization result

show_image(input_image)
