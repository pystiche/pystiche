import os
import random
import numpy as np
import torch
from pystiche.image import read_image
from pystiche.image.transforms import GrayscaleToBinary


def get_pystiche_root(file):
    return os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])


def print_sep_line():
    print("-" * 80)


def print_replication_info(title, url, author, year):
    info = (
        "This is the replication of the paper",
        f"'{title}'",
        url,
        "authored by",
        author,
        f"in {str(year)}",
    )
    print("\n".join(info))
    print_sep_line()


def make_reproducible(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device):
    if isinstance(device, torch.device):
        return device

    if isinstance(device, str):
        return torch.device(device)
    elif device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError("device should be a torch.device, a str, or None")


def get_input_image(starting_point, content_image=None, style_image=None):
    if isinstance(starting_point, torch.Tensor):
        return starting_point

    if starting_point not in ("content", "style", "random"):
        raise ValueError("starting_point should be 'content', 'style', or ''random")

    if starting_point == "content":
        if content_image is not None:
            return content_image.clone()
        raise ValueError("starting_point is 'content', but no content image is given")
    elif starting_point == "style":
        if style_image is not None:
            return style_image.clone()
        raise ValueError("starting_point is 'style', but no style image is given")
    elif starting_point == "random":
        if content_image is not None:
            return torch.rand_like(content_image)
        elif style_image is not None:
            return torch.rand_like(style_image)
        raise ValueError("starting_point is 'random', but no image is given")


def read_guides(root, image_file, device):
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    guide_folder = os.path.join(root, image_name)
    transform = GrayscaleToBinary()
    return {
        os.path.splitext(guide_file)[0]: transform(
            read_image(os.path.join(guide_folder, guide_file)).to(device)
        )
        for guide_file in os.listdir(guide_folder)
    }
