"""
Guided image optimization
=========================
"""


########################################################################################
# imports

from pystiche.demo import demo_images
from pystiche.enc import vgg19_multi_layer_encoder
from pystiche.image import guides_to_segmentation, show_image, write_image
from pystiche.loss import GuidedPerceptualLoss, PerceptualLoss
from pystiche.misc import get_input_image
from pystiche.misc.misc import get_device
from pystiche.ops import (
    FeatureReconstructionOperator,
    GramOperator,
    MultiLayerEncodingOperator,
    MultiRegionOperator,
)
from pystiche.optim import default_image_optim_loop

########################################################################################
# Preparation

device = get_device()
images = demo_images()
images.download()


########################################################################################
# Create the content and style loss and combine them into the optimization criterion

multi_layer_encoder = vgg19_multi_layer_encoder()

content_layer = "relu4_2"
content_encoder = multi_layer_encoder.extract_single_layer_encoder(content_layer)
content_weight = 1e0
content_loss = FeatureReconstructionOperator(
    content_encoder, score_weight=content_weight
)

style_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
style_weight = 1e3


def get_style_op(encoder, layer_weight):
    return GramOperator(encoder, score_weight=layer_weight)


style_loss = MultiLayerEncodingOperator(
    multi_layer_encoder, style_layers, get_style_op, score_weight=style_weight,
)


criterion = PerceptualLoss(content_loss, style_loss).to(device)
print(criterion)


########################################################################################
# load the content and style images and transfer them to the selected device
# the images are resized, since the stylization is memory intensive

size = 500

########################################################################################

content_image = images["lemgo_castle"].read(size=size, device=device)
show_image(content_image)


########################################################################################

style_image = images["segovia_fortress"].read(size=size, device=device)
show_image(style_image)


########################################################################################
# Set the target images for the content and style loss

criterion.set_content_image(content_image)
criterion.set_style_image(style_image)

starting_point = "content"
input_image = get_input_image(starting_point, content_image=content_image)

output_image = default_image_optim_loop(input_image, criterion, num_steps=500)

show_image(output_image)
write_image(output_image, "without_guidance.jpg")


########################################################################################
# Guided style loss


########################################################################################

content_guides = images["lemgo_castle"].guides.read(size=size, device=device)
content_segmentation = guides_to_segmentation(content_guides)
show_image(content_segmentation, title="Content segmentation")


########################################################################################

style_guides = images["segovia_fortress"].guides.read(size=size, device=device)
style_segmentation = guides_to_segmentation(style_guides)
show_image(style_segmentation, title="Style segmentation")


########################################################################################

regions = ("building", "sky")


def get_region_op(region, region_weight):
    return MultiLayerEncodingOperator(
        multi_layer_encoder, style_layers, get_style_op, score_weight=region_weight,
    )


style_loss = MultiRegionOperator(regions, get_region_op, score_weight=style_weight)

criterion = GuidedPerceptualLoss(content_loss, style_loss).to(device)
print(criterion)


########################################################################################
# Guided optimization

criterion.set_content_image(content_image)

for region in regions:
    criterion.set_style_guide(region, style_guides[region])
    criterion.set_style_image(region, style_image)
    criterion.set_content_guide(region, content_guides[region])

starting_point = "content"
input_image = get_input_image(starting_point, content_image=content_image)

output_image = default_image_optim_loop(input_image, criterion, num_steps=500)

# sphinx_gallery_thumbnail_number = 6
show_image(output_image)
write_image(output_image, "with_guidance.jpg")
