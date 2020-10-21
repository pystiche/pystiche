import time
from collections import OrderedDict
from os import path

import torch
from torch import hub, nn
from torch.nn.functional import interpolate

import pystiche
from pystiche import demo, enc, loss, ops, optim
from pystiche.image import show_image, transforms
from pystiche.misc import get_device

print(f"I'm working with pystiche=={pystiche.__version__}")

device = get_device()
print(f"I'm working with {device}")

########################################################################################

images = demo.images()
images.download()
size = 500

########################################################################################

style_image = images["paint"].read(device=device)
# show_image(style_image)

########################################################################################


class Interpolate(nn.Module):
    def __init__(
        self, scale_factor=None, mode="nearest",
    ):
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


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv(channels, channels, kernel_size=3)
        self.conv2 = Conv(channels, channels, kernel_size=3, activation=False)

    def forward(self, input):
        output = self.conv2(self.conv1(input))
        return output + input


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            transforms.FloatToUint8Range(),
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
            transforms.Uint8ToFloatRange(),
        )

    def forward(self, input):
        return self.decoder(self.encoder(input))


transformer = Transformer().to(device)
print(transformer)

########################################################################################

multi_layer_encoder = enc.vgg16_multi_layer_encoder()

content_layer = "relu4_2"
content_encoder = multi_layer_encoder.extract_encoder(content_layer)
content_weight = 1e5
content_loss = ops.FeatureReconstructionOperator(
    content_encoder, score_weight=content_weight
)

style_layers = ("relu1_2", "relu2_2", "relu3_3", "relu4_3")
style_weight = 1e10
style_loss = ops.MultiLayerEncodingOperator(
    multi_layer_encoder,
    style_layers,
    lambda encoder, layer_weight: ops.GramOperator(encoder, score_weight=layer_weight),
    layer_weights="sum",
    score_weight=style_weight,
)

criterion = loss.PerceptualLoss(content_loss, style_loss).to(device)
print(criterion)

########################################################################################


class OptionalGrayscaleToFakeGrayscale(transforms.Transform):
    def forward(self, input):
        num_channels = input.size()[0]
        if num_channels == 1:
            return input.repeat(3, 1, 1)
        elif num_channels == 3:
            return input
        else:
            raise RuntimeError


# preprocessor = transforms.FloatToUint8Range().to(device)
# postprocessor = transforms.Uint8ToFloatRange().to(device)


def train(
    root="~/datasets/coco/train2014", image_size=256, batch_size=4, epochs=2,
):
    from torch.utils.data import DataLoader

    from pystiche.data import ImageFolderDataset

    transform = transforms.ComposedTransform(
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        OptionalGrayscaleToFakeGrayscale(),
    )
    dataset = ImageFolderDataset(root, transform=transform)
    image_loader = DataLoader(dataset, batch_size=batch_size)

    criterion.set_style_image(style_image.repeat(batch_size, 1, 1, 1))

    def criterion_update_fn(input_image, criterion) -> None:
        criterion.set_content_image(input_image)

    optim.default_transformer_epoch_optim_loop(
        image_loader,
        transformer.train(),
        criterion,
        criterion_update_fn,
        epochs=epochs,
        # logger=demo.logger(),
    )


########################################################################################

# FIXME
use_pretrained_transformer = False
checkpoint = "example_transformer.pth"

if use_pretrained_transformer:
    if path.exists(checkpoint):
        state_dict = torch.load(checkpoint)
    else:
        url = "https://download.pystiche.org/models/example_transformer.pth"
        state_dict = hub.load_state_dict_from_url(url)

    transformer.load_state_dict(state_dict)
else:
    train()

    state_dict = OrderedDict(
        [
            (name, parameter.detach().cpu())
            for name, parameter in transformer.state_dict().items()
        ]
    )
    torch.save(state_dict, "example_transformer.pth")

########################################################################################

content_image = images["bird1"].read(device=device)
# show_image(content_image)

########################################################################################


transformer.eval()

start = time.time()

with torch.no_grad():
    output_image = transformer(content_image)

stop = time.time()

# sphinx_gallery_thumbnail_number = 3
show_image(output_image, title="Output image")

########################################################################################

print(f"The stylization took {(stop - start) * 1e3:.0f} milliseconds.")
