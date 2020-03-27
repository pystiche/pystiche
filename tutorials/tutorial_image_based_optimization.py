"""
NST via image-based optimization
================================
"""


###############################################################################
# imports

import torch
from torch import optim
from pystiche.image import show_image, write_image
from pystiche.enc import vgg19_multi_layer_encoder
from pystiche.ops import MSEEncodingOperator, GramOperator, MultiLayerEncodingOperator
from pystiche.loss import PerceptualLoss
from pystiche.demo import demo_images


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
# load the content and style images and transfer them to the selected device
# the images are resized, since the stylization is memory intensive

size = 500
images = demo_images()
content_image = images["dancing"].read(size=size, device=device)
style_image = images["picasso"].read(size=size, device=device)
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
# Create the optimizer that performs the stylization

optimizer = optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


###############################################################################
# .. note::
#   To avoid boilerplate code, you can achieve the same behavior with
#   :func:`~pystiche.optim.optim.default_image_optimizer`::
#
#     from pystiche.optim import default_image_optimizer
#
#     optimizer = default_image_optimizer(input_image)


###############################################################################
# Run the stylization

num_steps = 500
for step in range(1, num_steps + 1):

    def closure():
        optimizer.zero_grad()
        loss = criterion(input_image)
        loss.backward()

        if step % 50 == 0:
            print(f"Step {step}")
            print()
            print(loss.aggregate(1))
            print("-" * 80)

        return loss

    optimizer.step(closure)


###############################################################################
# .. note::
#   To avoid boilerplate code, you can achieve the same behavior with
#   :func:`~pystiche.optim.optim.default_image_optim_loop`::
#
#     from pystiche.optim import default_image_optim_loop
#
#     default_image_optim_loop(
#         input_image, criterion, optimizer=optimizer, num_steps=num_steps
#     )
#
#   If you do not pass ``optimizer``
#   :func:`~pystiche.optim.optim.default_image_optimizer` is used.


###############################################################################
# Show the stylization result

show_image(input_image)
