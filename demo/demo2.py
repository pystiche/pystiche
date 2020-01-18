from os import path
from collections import OrderedDict
import torch
from torch import optim
from pystiche.image import read_image, write_image, extract_aspect_ratio
from pystiche.enc import vgg19_encoder
from pystiche.ops import MSEEncodingLoss, GramLoss, MultiLayerEncodingComparisonOperator
from pystiche.loss import MultiOperatorLoss
from pystiche.pyramid import ImagePyramid

# load the encoder used to create the feature maps for the NST
multi_layer_encoder = vgg19_encoder()

# create the content loss
content_layer = "relu_4_2"
content_encoder = multi_layer_encoder.extract_single_layer_encoder(content_layer)
content_weight = 1e0
content_loss = MSEEncodingLoss(content_encoder, score_weight=content_weight)

# create the style loss
style_layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
style_weight = 1e4
style_loss = MultiLayerEncodingComparisonOperator(
    GramLoss,
    multi_layer_encoder,
    style_layers,
    score_weight=style_weight,
)

# combine the content and style loss into the optimization criterion
criterion = MultiOperatorLoss(
    OrderedDict([("content_loss", content_loss), ("style_loss", style_loss)])
)

# make this demo device-agnostic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = criterion.to(device)

edge_sizes = (500, 700)
num_steps = (500, 200)
pyramid = ImagePyramid(edge_sizes, num_steps, resize_targets=(criterion,))

# adapt these paths to fit your use case
# you can find a download script for some frequently used images in
# $PYSTICHE_PROJECT_ROOT/images
content_file = path.expanduser(path.join("../milky_way_over_lake_alberta.jpg"))
style_file = path.expanduser(path.join("../starry_night.jpg"))
output_file = "../pystiche_demo.jpg"

# load the content and style images and transfer them to the selected device
content_image = read_image(content_file, device=device)
style_image = read_image(style_file, device=device)

# resize
resize = pyramid[-1].resize_image
content_image = resize(content_image)
style_image = resize(style_image)

# set the target images for the content and style loss
content_loss.set_target_image(content_image)
style_loss.set_target_image(style_image)

# start teh stylization from the content image
input_image = content_image.clone()
# uncomment the following line if you want to start from a white noise image instead
# input_image = torch.rand_like(content_image)

aspect_ratio = extract_aspect_ratio(input_image)


# create optimizer that performs the stylization
def get_optimizer(input_image):
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


# run the stylization
for level in pyramid:
    input_image = level.resize_image(input_image, aspect_ratio=aspect_ratio)
    optimizer = get_optimizer(input_image)

    for step in level:

        def closure():
            optimizer.zero_grad()
            loss = criterion(input_image)
            loss.backward()

            if step % 20 == 0:
                print(loss["content_loss"].item(), loss["style_loss"].item())

            return loss

        optimizer.step(closure)

# save the stylized image
write_image(input_image, output_file)
