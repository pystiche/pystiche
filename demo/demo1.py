from os import path
from collections import OrderedDict
import torch
from torch import optim
from pystiche.image import read_image, write_image
from pystiche.enc import vgg19_encoder
from pystiche.ops import MSEEncodingOperator, GramOperator, MultiLayerEncodingOperator
from pystiche.loss import MultiOperatorLoss

# load the encoder used to create the feature maps for the NST
multi_layer_encoder = vgg19_encoder()

# create the content loss
content_layer = "relu_4_2"
content_encoder = multi_layer_encoder[content_layer]
content_weight = 1e0
content_loss = MSEEncodingOperator(content_encoder, score_weight=content_weight)

# create the style loss
style_layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
style_weight = 1e4
style_loss = MultiLayerEncodingOperator(
    GramOperator, multi_layer_encoder, style_layers, score_weight=style_weight
)

# combine the content and style loss into the optimization criterion
criterion = MultiOperatorLoss(
    OrderedDict([("content_loss", content_loss), ("style_loss", style_loss)])
)

# make this demo device-agnostic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = criterion.to(device)

# adapt these paths to fit your use case
content_file = path.join("path", "to", "content", "image.jpg")
style_file = path.join("path", "to", "style", "image.jpg")
output_file = "pystiche_demo1.jpg"

# load the content and style images and transfer them to the selected device
# the images are resized, since the stylization is memory intensive
size = 500
content_image = read_image(content_file, device=device, size=size)
style_image = read_image(style_file, device=device, size=size)

# set the target images for the content and style loss
content_loss.set_target_image(content_image)
style_loss.set_target_image(style_image)

# start the stylization from the content image
input_image = content_image.clone()
# uncomment the following line if you want to start from a white noise image instead
# input_image = torch.rand_like(content_image)

# create optimizer that performs the stylization
optimizer = optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)

# run the stylization
num_steps = 500
for step in range(num_steps):

    def closure():
        optimizer.zero_grad()
        loss = criterion(input_image)
        loss.backward()

        if step % 20 == 0:
            print(loss)
            print("-" * 80)

        return loss

    optimizer.step(closure)

# save the stylized image
write_image(input_image, output_file)
