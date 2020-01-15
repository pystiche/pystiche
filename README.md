# pystiche

The pystiche project is a free, open-source framework for Neural Style Transfer (NST) algorithms. Roughly speaking, pystiche relates to NST algorithms as PyTorch relates to deep learning.

The name of the project is a pun on [_pastiche_](https://en.wikipedia.org/wiki/Pastiche) meaning:

> A pastiche is a work of visual art [...] that imitates the style or character of the work of one or more other artists. Unlike parody, pastiche celebrates, rather than mocks, the work it imitates.

# Installation

`pystiche` requires `python 3.6` or later. Make sure you install `torch` and `torchvision` from the [official sources](https://pytorch.org/get-started/locally/) in order to avoid any installation errors.

```bash
git clone https://github.com/pmeier/pystiche
pip install pystiche/
```

# Demo 1

```python
from os import path
import torch
from torch import optim
from pystiche.image import read_image, write_image
from pystiche.enc import vgg19_encoder
from pystiche.ops import (
    DirectEncodingComparisonOperator,
    GramEncodingComparisonOperator,
)
from pystiche.loss import MultiOperatorLoss, MultiOperatorEncoder

# adapt these paths to fit your use case
# you can find a download script for some frequently used images in
# $PYSTICHE_PROJECT_ROOT/images
content_file = path.join("path", "to", "content_image")
style_file = path.join("path", "to", "style_image")
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
encoder = MultiOperatorEncoder(encoder)

# create the content operator
name = "Content loss"
layers = ("relu_4_2",)
score_weight = 1e0
content_operator = DirectEncodingComparisonOperator(
    encoder, layers, name=name, score_weight=score_weight
)

# create the style operator
name = "Style loss"
layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
score_weight = 1e4
style_operator = GramEncodingComparisonOperator(
    encoder, layers, name=name, score_weight=score_weight
)

# create the image optimizer and transfer it to the selected device
criterion = MultiOperatorLoss(content_operator, style_operator).to(device)

# set the target images for the operators
content_operator.set_target(content_image)
style_operator.set_target(style_image)

# create starting point for the stylization
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
        return loss

# save the stylized image
write_image(input_image, output_file)
```

# Demo 1

```python
from os import path
import torch
from torch import optim
from pystiche.image import read_image, write_image
from pystiche.enc import vgg19_encoder
from pystiche.ops import (
    DirectEncodingComparisonOperator,
    GramEncodingComparisonOperator,
)
from pystiche.loss import MultiOperatorLoss, MultiOperatorEncoder
from pystiche.pyramid import ImagePyramid

# adapt these paths to fit your use case
# you can find a download script for some frequently used images in
# $PYSTICHE_PROJECT_ROOT/images
content_file = path.join("path", "to", "content_image")
style_file = path.join("path", "to", "style_image")
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
encoder = MultiOperatorEncoder(encoder)

# create the content operator
name = "Content loss"
layers = ("relu_4_2",)
score_weight = 1e0
content_operator = DirectEncodingComparisonOperator(
    encoder, layers, name=name, score_weight=score_weight
)

# create the style operator
name = "Style loss"
layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
score_weight = 1e4
style_operator = GramEncodingComparisonOperator(
    encoder, layers, name=name, score_weight=score_weight
)

# create the image optimizer and transfer it to the selected device
criterion = MultiOperatorLoss(content_operator, style_operator).to(device)

edge_sizes = (500, 800)
num_steps = (500, 200)
pyramid = ImagePyramid(edge_sizes, num_steps)

# set the target images for the operators
content_operator.set_target(content_image)
style_operator.set_target(style_image)

# create starting point for the stylization
input_image = content_image.clone()
# uncomment the following line if you want to start from a white noise image instead
# input_image = torch.rand_like(content_image)

pyramid.resize_targets = (criterion, input_image)

# create optimizer that performs the stylization
optimizer = optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)

# run the stylization
for level in pyramid:
    for step in level:
        def closure():
            optimizer.zero_grad()
            loss = criterion(input_image)
            loss.backward()
            return loss
        
        optimizer.step()

# save the stylized image
write_image(input_image, output_file)
```

