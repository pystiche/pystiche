"""
Model optimization
==================

This example showcases how an NST based on model optimization can be performed in
``pystiche``. It closely follows the
`official PyTorch example <https://github.com/pytorch/examples/tree/master/fast_neural_style>`_
which in turn is based on :cite:`JAL2016`.
"""


########################################################################################
# We start this example by importing everything we need and setting the device we will
# be working on.

import time
from collections import OrderedDict
from os import path

import torch
from torch import hub, nn
from torch.nn.functional import interpolate

import pystiche
from pystiche import demo, enc, loss, ops, optim
from pystiche.image import show_image
from pystiche.misc import get_device

print(f"I'm working with pystiche=={pystiche.__version__}")

device = get_device()
print(f"I'm working with {device}")


########################################################################################
# Transformer
# -----------

# In contrast to image optimization, for model optimization we need to define a
# transformer that, after it is trained, performs the stylization. In general different
# architectures are possible (:cite:`JAL2016,ULVL2016`). For this example we use an
# encoder-decoder architecture.
#
# Before we define the transformer, we create some helper modules to reduce the clutter.


########################################################################################
# In the decoder we need to upsample the image. While it is possible to achieve this
# with a :class:`~torch.nn.ConvTranspose2d`, it was found that traditional upsampling
# followed by a standard convolution produces fewer artifacts :cite:`JAL2016`. Thus,
# we create an module that wraps :func:`~torch.nn.functional.interpolate`.


class Interpolate(nn.Module):
    def __init__(self, scale_factor=1.0, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        return interpolate(input, scale_factor=self.scale_factor, mode=self.mode,)

    def extra_repr(self):
        extras = []
        if self.scale_factor:
            extras.append(f"scale_factor={self.scale_factor}")
        if self.mode != "nearest":
            extras.append(f"mode={self.mode}")
        return ", ".join(extras)


########################################################################################
# For the transformer architecture we will be using, we need to define a convolution
# module with some additional capabilities. In particular, it needs to be able to
# - optionally upsample the input,
# - pad the input in order for the convolution to be size-preserving,
# - optionally normalize the output, and
# - optionally pass the output through an activation function.
#
# .. note::
#
#   Instead of :class:`~torch.nn.BatchNorm2d` we use :class:`~torch.nn.InstanceNorm2d`
#   to normalize the output since it gives better results for NST :cite:`UVL2016`.


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        upsample=False,
        norm=True,
        activation=True,
    ):
        super().__init__()
        self.upsample = Interpolate(scale_factor=stride) if upsample else None
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1 if upsample else stride
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if norm else None
        self.activation = nn.ReLU() if activation else None

    def forward(self, input):
        if self.upsample:
            input = self.upsample(input)

        output = self.conv(self.pad(input))

        if self.norm:
            output = self.norm(output)
        if self.activation:
            output = self.activation(output)

        return output


########################################################################################
# It is common practice to append a few residual blocks after the initial convolutions
# to the encoder to enable it to learn more descriptive features.


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv(channels, channels, kernel_size=3)
        self.conv2 = Conv(channels, channels, kernel_size=3, activation=False)

    def forward(self, input):
        output = self.conv2(self.conv1(input))
        return output + input


########################################################################################
# It can be useful for the training to transform the input into another value range,
# for example from :math:`\closedinterval{0}{1}` to :math:`\closedinterval{0}{255}`.


class FloatToUint8Range(nn.Module):
    def forward(self, input):
        return input * 255.0


class Uint8ToFloatRange(nn.Module):
    def forward(self, input):
        return input / 255.0


########################################################################################
# Finally, we can put all pieces together.


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            Conv(3, 32, kernel_size=9),
            Conv(32, 64, kernel_size=3, stride=2),
            Conv(64, 128, kernel_size=3, stride=2),
            Residual(128),
            Residual(128),
            Residual(128),
            Residual(128),
            Residual(128),
        )
        self.decoder = nn.Sequential(
            Conv(128, 64, kernel_size=3, stride=2, upsample=True),
            Conv(64, 32, kernel_size=3, stride=2, upsample=True),
            Conv(32, 3, kernel_size=9, norm=False, activation=False),
        )

        self.preprocessor = FloatToUint8Range()
        self.postprocessor = Uint8ToFloatRange()

    def forward(self, input):
        input = self.preprocessor(input)
        output = self.decoder(self.encoder(input))
        return self.postprocessor(output)


transformer = Transformer().to(device)
print(transformer)


########################################################################################
# Perceptual loss
# ---------------
#
# Although model optimization is a different paradigm, the perceptual loss is the same
# as for image optimization.
#
# .. note::
#
#  In some implementations, such as the PyTorch example and :cite:`JAL2016`, one can
#  observe that the :func:`~pystiche.gram_matrix`, used as style representation, is not
#  only normalized by the height and width of the feature map, but also by the number
#  of channels. If used togehter with a :func:`~torch.nn.functional.mse_loss`, the
#  normalization is performed twice. While this is unintended, it affects the training.
#  In order to keep the other hyper parameters on par with the PyTorch example, we also
#  adopt this change here.

multi_layer_encoder = enc.vgg16_multi_layer_encoder()

content_layer = "relu2_2"
content_encoder = multi_layer_encoder.extract_encoder(content_layer)
content_weight = 1e5
content_loss = ops.FeatureReconstructionOperator(
    content_encoder, score_weight=content_weight
)


class GramOperator(ops.GramOperator):
    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        repr = super().enc_to_repr(enc)
        num_channels = repr.size()[1]
        return repr / num_channels


style_layers = ("relu1_2", "relu2_2", "relu3_3", "relu4_3")
style_weight = 1e10
style_loss = ops.MultiLayerEncodingOperator(
    multi_layer_encoder,
    style_layers,
    lambda encoder, layer_weight: GramOperator(encoder, score_weight=layer_weight),
    layer_weights="sum",
    score_weight=style_weight,
)

criterion = loss.PerceptualLoss(content_loss, style_loss).to(device)
print(criterion)


########################################################################################
# Training
# --------
#
# In a first step we load the style image that will be used to train the
# ``transformer``.

images = demo.images()
size = 500

style_image = images["paint"].read(size=size, device=device)
show_image(style_image)


########################################################################################
# The training of the ``transformer`` is performed similar to other models in PyTorch.
# In every optimization step a batch of content images is drawn from a dataset, which
# serve as input for the transformer as well as ``content_image`` for the
# ``criterion``. While the ``style_image`` only has to be set once, the
# ``content_image`` has to be reset in every iteration step.
#
# While this can be done with a boilerplate optimization loop, ``pystiche`` provides
# :func:`~pystiche.optim.multi_epoch_model_optimization` that handles the above for you.
#
# .. note::
#
#   If the ``criterion`` is a :class:`~pystiche.loss.PerceptualLoss`, as is the case
#   here, the update of the ``content_image`` is performed automatically. If that is
#   not the case or you need more complex update behavior, you need to specify a
#   ``criterion_update_fn``.
#
# .. note::
#
#   If you do not specify an ``optimizer``, the
#   :func:`~pystiche.optim.default_model_optimizer`, i.e.
#   :class:`~torch.optim.Adam` is used.


def train(
    transformer, dataset, batch_size=4, epochs=2,
):
    if dataset is None:
        raise RuntimeError(
            "You forgot to define a dataset. For example, "
            "you can use any image dataset from torchvision.datasets."
        )

    from torch.utils.data import DataLoader

    image_loader = DataLoader(dataset, batch_size=batch_size)

    criterion.set_style_image(style_image)

    return optim.multi_epoch_model_optimization(
        image_loader,
        transformer.train(),
        criterion,
        epochs=epochs,
        logger=demo.logger(),
    )


########################################################################################
# Depending on the dataset and your setup the training can take a couple of hours. To
# avoid this, we provide transformer weights that were trained with the scheme above.
#
# .. note::
#
#   If you want to perform the training yourself, set
#   ``use_pretrained_transformer=False``. If you do, you also need to replace
#   ``dataset = None`` below with the dataset you want to train on.
#
# .. note::
#
#   The weights of the provided transformer were trained with the
#   `2014 training images <http://images.cocodataset.org/zips/train2014.zip>`_ of the
#   `COCO dataset <https://cocodataset.org/>`_. The training was performed for
#   ``num_epochs=2`` and ``batch_size=4``. Each image was center-cropped to
#   ``256 x 256`` pixels.

use_pretrained_transformer = True
checkpoint = "example_transformer.pth"

if use_pretrained_transformer:
    if path.exists(checkpoint):
        state_dict = torch.load(checkpoint)
    else:
        url = "https://download.pystiche.org/models/example_transformer.pth"
        state_dict = hub.load_state_dict_from_url(url)

    transformer.load_state_dict(state_dict)
else:
    dataset = None
    transformer = train(transformer, dataset)

    state_dict = OrderedDict(
        [
            (name, parameter.detach().cpu())
            for name, parameter in transformer.state_dict().items()
        ]
    )
    torch.save(state_dict, checkpoint)


########################################################################################
# Neural Style Transfer
# ---------------------
#
# In order to perform the NST, we load an image we want to stylize.

input_image = images["bird1"].read(size=size, device=device)
show_image(input_image)


########################################################################################
# After the transformer is trained we can now perform an NST with a single forward pass.
# To do this, the ``transformer`` is simply called with the ``input_image``.

transformer.eval()

start = time.time()

with torch.no_grad():
    output_image = transformer(input_image)

stop = time.time()

# sphinx_gallery_thumbnail_number = 3
show_image(output_image, title="Output image")


########################################################################################
# Compared to NST via image optimization, the stylization is performed multiple orders
# of magnitudes faster. Given capable hardware, NST via model optimization enables
# real-time stylization for example of a video feed.

print(f"The stylization took {(stop - start) * 1e3:.0f} milliseconds.")
