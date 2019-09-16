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

# Demo

```python
from os import path
import torch
from pystiche.image import read_image, write_image
from pystiche.encoding import vgg19_encoder
from pystiche.nst import (
    MultiOperatorEncoder,
    DirectEncodingComparisonOperator,
    GramEncodingComparisonOperator,
    ImageOptimizer,
)

# adapt these paths to fit your use case
# you can find a download script for some frequently used images in
# $PYSTICHE_PROJECT_ROOT/images
content_file = path.join("path", "to", "content_image")
style_file = path.join("path", "to", "style_image")
output_file = "pystiche_demo.jpg"

# make this demo device-agnostic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the content and style images and transfer them to the selected device
content_image = read_image(content_file).to(device)
style_image = read_image(style_file).to(device)

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
nst = ImageOptimizer(content_operator, style_operator).to(device)

# set the target images for the operators
content_operator.set_target(content_image)
style_operator.set_target(style_image)

# create starting point for the stylization
input_image = content_image.clone()
# uncomment the following line if you want to start from a white noise image instead
# input_image = torch.rand_like(content_image)

# run the stylization
num_steps = 500
output_image = nst(input_image, num_steps)

# save the stylized image
write_image(output_image, output_file)
```

# Replication

The `pystiche` project features a replication study for NST papers. You can find the scripts in `$PYSTICHE_ROOT/replication`. Currently the following papers are pre-implemented:

- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys, Ecker and Bethge
- [Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis](https://ieeexplore.ieee.org/document/7780641) by Li and Wand
- [Controlling Perceptual Factors in Neural Style Transfer](http://openaccess.thecvf.com/content_cvpr_2017/papers/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.pdf) by Gatys et. al.

# Webapp

The `pystiche` project features a webapp as frontend, which performs the demo code. To start it

```bash
cd $PYSTICHE_ROOT/webapp
python3 manage.py runserver 8080
```

Afterwards click on the link displayed in the console or type `localhost:8080` as URL in your favorite browser.
