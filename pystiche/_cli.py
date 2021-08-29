import argparse
import functools
import os.path
import pathlib
import sys
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import torch

import pystiche
from pystiche import demo, enc, loss, optim
from pystiche.data import LocalImage
from pystiche.data.collections._core import _Image
from pystiche.image import extract_image_size, write_image


def sys_exit(fn: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as error:
            print(error, file=sys.stderr)
            status = 1
        else:
            status = 0

        sys.exit(status)

    return wrapper


@sys_exit
def main(raw_args: Optional[List[str]] = None) -> None:
    args = parse_args(raw_args)
    config = make_config(args)

    config.perceptual_loss.set_content_image(config.content_image)
    config.perceptual_loss.set_style_image(config.style_image)

    output_image = optim.image_optimization(
        config.input_image, config.perceptual_loss, num_steps=config.num_steps
    )
    write_image(output_image, config.output_image)


def make_config(args: argparse.Namespace) -> SimpleNamespace:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    content_image = load_image(
        args.content_image, size=args.content_size, device=device
    )
    style_image = load_image(args.style_image, size=args.style_size, device=device)
    input_image = make_input_image(args.starting_point, content_image=content_image)
    output_image = make_output_image(args.output_image)

    mle = make_multi_layer_encoder(args.multi_layer_encoder).to(device)

    perceptual_loss = make_perceptual_loss(args, mle).to(device)

    return SimpleNamespace(
        content_image=content_image,
        style_image=style_image,
        input_image=input_image,
        output_image=output_image,
        perceptual_loss=perceptual_loss,
        num_steps=args.num_steps,
    )


def load_image(
    file_or_name: str, *, size: Union[int, Tuple[int, int]], device: torch.device
) -> torch.Tensor:
    file = os.path.abspath(os.path.expanduser(file_or_name))
    images = demo.images()
    image: _Image
    if os.path.exists(file):
        image = LocalImage(file)
    elif file_or_name in images:
        image = images[file_or_name]
    else:
        raise ValueError(
            f"'{file_or_name}' is neither an image file nor a name of a demo image."
        )

    return image.read(size=size, device=device)


def make_input_image(
    starting_point: str, *, content_image: torch.Tensor
) -> torch.Tensor:
    if starting_point == "content":
        return content_image.clone()
    elif starting_point == "random":
        return torch.rand_like(content_image)
    else:
        return load_image(
            starting_point,
            size=extract_image_size(content_image),
            device=content_image.device,
        )


def make_output_image(output_image: Optional[str]) -> str:
    default_path = pathlib.Path().resolve()
    default_name = f"pystiche_{datetime.now():%Y%m%d%H%M%S}.jpg"

    if output_image is None:
        return str(default_path / default_name)

    path = pathlib.Path(output_image)
    if path.is_dir():
        return str(path / default_name)

    return output_image


def make_multi_layer_encoder(mle_str: str) -> enc.ModelMultiLayerEncoder:
    return cast(
        enc.ModelMultiLayerEncoder,
        getattr(enc, f"{mle_str.lower()}_multi_layer_encoder")(),
    )


LOSS_CLSS = {
    name.lower().replace("loss", ""): cls
    for name, cls in loss.__dict__.items()
    if isinstance(cls, type)
    and issubclass(cls, loss.ComparisonLoss)
    and cls is not loss.ComparisonLoss
}


def make_loss(
    loss_str: str, layers_str: str, score_weight: float, mle: enc.MultiLayerEncoder
) -> Union[loss.ComparisonLoss, loss.MultiLayerEncodingLoss]:
    loss_str = loss_str.lower().replace("_", "").replace("-", "")
    if loss_str not in LOSS_CLSS.keys():
        raise ValueError(f"Unknown loss class {loss_str}.")

    loss_cls = LOSS_CLSS[loss_str]

    layers = [layer.strip() for layer in layers_str.split(",")]
    layers = sorted(layer for layer in layers if layer)
    for layer in layers:
        if layer not in mle:
            raise ValueError(f"Unknown layer {layer} in MLE.")

    if len(layers) == 1:
        return loss_cls(
            encoder=mle.extract_encoder(layers[0]), score_weight=score_weight
        )
    else:
        return loss.MultiLayerEncodingLoss(
            mle,
            layers,
            lambda encoder, layer_weight: loss_cls(
                encoder=encoder, score_weight=layer_weight
            ),
            score_weight=score_weight,
        )


def make_perceptual_loss(
    args: argparse.Namespace, mle: enc.MultiLayerEncoder,
) -> loss.PerceptualLoss:
    content_loss = make_loss(
        args.content_loss, args.content_layers, args.content_weight, mle
    )
    style_loss = make_loss(args.style_loss, args.style_layers, args.style_weight, mle)
    return loss.PerceptualLoss(content_loss, style_loss)


def parse_args(raw_args: Optional[List[str]] = None) -> argparse.Namespace:
    if raw_args is None:
        raw_args = sys.argv[1:]

    parser = make_parser()
    return parser.parse_args(raw_args)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--multi-layer-encoder", "--mle", type=str, default="vgg19",
    )

    parser.add_argument(
        "--content-loss", "--cl", type=str, default="FeatureReconstruction",
    )
    parser.add_argument(
        "--content-layers", "--cla", type=str, default="relu4_2",
    )
    parser.add_argument(
        "--content-weight", "--cw", type=float, default=1e0,
    )
    parser.add_argument(
        "--content-size", "--cs", type=int, default=500,
    )

    parser.add_argument(
        "--style-loss", "--sl", type=str, default="Gram",
    )
    parser.add_argument(
        "--style-layers",
        "--sla",
        type=str,
        default="relu1_1,relu2_1,relu3_1,relu4_1,relu5_1",
    )
    parser.add_argument(
        "--style-size", "--ss", type=int, default=500,
    )
    parser.add_argument(
        "--style-weight", "--sw", type=float, default=1e3,
    )

    parser.add_argument("-o", "--output-image", type=str)
    parser.add_argument(
        "-n", "--num-steps", type=int, default=500,
    )
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument(
        "-s", "--starting-point", type=str, default="content",
    )

    parser.add_argument("content_image", type=str)
    parser.add_argument("style_image", type=str)

    parser.add_argument(
        "-V", "--version", action="version", version=pystiche.__version__,
    )

    return parser
