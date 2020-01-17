from os import path
import torch
from torch import optim
from pystiche.image import read_image, write_image, show_image
from pystiche.enc import vgg19_encoder
from pystiche.ops import (
    DirectEncodingComparisonOperator,
    GramEncodingComparisonOperator,
    TotalVariationRegularizationOperator,
    MultiLayerEncodingComparisonOperator,
DirectComparisonOperator,
TotalVariationEncodingRegularizationOperator
)
from pystiche.loss import MultiOperatorLoss

# adapt these paths to fit your use case
# you can find a download script for some frequently used images in
# $PYSTICHE_PROJECT_ROOT/images
content_file = path.expanduser(path.join("~", "Pictures", "portrait.jpg"))
style_file = path.expanduser(path.join("~", "Pictures", "style.png"))
output_file = "pystiche_demo.jpg"

# make this demo device-agnostic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the content and style images and transfer them to the selected device
# the images are resized, since the stylization is memory intensive
size = 500
content_image = read_image(content_file, device=device, size=size)
style_image = read_image(style_file, device=device, size=size)

# load the encoder used to create the feature maps for the NST
encoder = vgg19_encoder()
# wrap it into an MultiOperatorEncoder() to avoid calculating the feature maps multiple times
# during a single step, if multiple operators use the same encoder
# encoder = MultiOperatorEncoder(encoder)
# print(encoder)

encoder = encoder.eval()

# create the content operator
content_layer = "relu_4_2"
content_weight = 1e0
content_loss = DirectEncodingComparisonOperator(
    encoder, content_layer, score_weight=content_weight
)


from typing import Callable, Collection, Union, Sequence, Iterator
from torch import nn
from pystiche.enc import Encoder
from pystiche.ops import (
    EncodingRegularizationOperator,
    EncodingComparisonOperator,
    Operator,
)


style_layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
style_weight = 1e4
style_loss = MultiLayerEncodingComparisonOperator(
    lambda layer, layer_weight: GramEncodingComparisonOperator(
        encoder, layer, score_weight=layer_weight
    ),
    style_layers,
    score_weight=style_weight,
)


# criterion = TotalVariationEncodingRegularizationOperator(encoder, "relu_2_2", score_weight=1e6).to(device)

# create the image optimizer and transfer it to the selected device
criterion = MultiOperatorLoss(content_loss, style_loss).to(device)
# criterion = MultiOperatorLoss(content_loss).to(device)
# print(criterion)

content_loss.set_target_image(content_image)
style_loss.set_target_image(style_image)

# criterion = style_loss.to(device)
# criterion.set_target_image(style_image)

# set the target images for the operators


# create starting point for the stylization
input_image = content_image.clone()
# uncomment the following line if you want to start from a white noise image
# rather than the content image
# input_image = torch.rand_like(content_image)

# create optimizer that performs the stylization
optimizer = optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)

# run the stylization
num_steps = 200
for step in range(num_steps):

    def closure():
        optimizer.zero_grad()
        loss = criterion(input_image)

        # print(torch.norm(input_image).item())
        print(loss["0"], loss["1"])
        # print("-" * 50)

        # print(loss.item())

        loss.backward()

        # print(torch.norm(input_image.grad))


        return loss


    optimizer.step(closure)

# save the stylized image
write_image(input_image, output_file)
