from os import path
import itertools
from pystiche.image import read_image as _read_image, write_image
from pystiche.papers import GatysEckerBethge2015NST
from pystiche.cuda import abort_if_cuda_memory_exausts
import utils

# Since no information is supplied in the paper, hyperparameters are taken from
# Johnsons implementation (https://github.com/jcjohnson/neural-style)

# https://github.com/jcjohnson/neural-style/blob/07c4b8299f8fbdafec0c514fc820ff1d7ff62e46/neural_style.lua#L25
NUM_STEPS = 1000

# https://github.com/jcjohnson/neural-style/blob/07c4b8299f8fbdafec0c514fc820ff1d7ff62e46/neural_style.lua#L17
MAX_EDGE_LENGTH = 512


def read_image(file):
    return _read_image(file, size=MAX_EDGE_LENGTH, edge="long")


@abort_if_cuda_memory_exausts
def figure_2(source_folder, replication_folder, device, impl_params=False):
    content_file = path.join(source_folder, "praefcke__tuebingen_neckarfront.jpg")
    content_image = read_image(content_file).to(device)

    class StyleImage:
        def __init__(self, label, file, weight_ratio):
            self.label = label
            self.data = read_image(path.join(source_folder, file)).to(device)
            self.weight_ratio = weight_ratio

    params = "implementation" if impl_params else "paper"
    style_images = (
        StyleImage(
            "B", "turner__shipwreck_of_the_minotaur.jpg", 50e-3 if impl_params else 1e-3
        ),
        StyleImage("C", "van_gogh__starry_night.jpg", 50e-3 if impl_params else 1e-3),
        StyleImage("D", "munch__the_scream.jpg", 50e-3 if impl_params else 1e-3),
        StyleImage(
            "E", "picasso__figure_dans_un_fauteuil.jpg", 50e-3 if impl_params else 1e-3
        ),
        StyleImage(
            "F", "kandinsky__composition_vii.jpg", 50e-3 if impl_params else 1e-3
        ),
    )

    nst = GatysEckerBethge2015NST(impl_params).to(device)
    nst.content_loss.set_target(content_image)
    content_score_weight = nst.content_loss.score_weight
    for style_image in style_images:
        print(f"Replicating Figure 2 {style_image.label} with {params} parameters")
        nst.content_loss.score_weight = content_score_weight / style_image.weight_ratio
        nst.style_loss.set_target(style_image.data)

        utils.make_reproducible()
        input_image = utils.get_input_image("random", content_image)
        output_image = nst(input_image, NUM_STEPS, quiet=True)

        output_file = path.join(
            replication_folder, "fig_2__{}.jpg".format(style_image.label)
        )
        print(f"Saving result to {output_file}")
        write_image(output_image, output_file)


@abort_if_cuda_memory_exausts
def figure_3(source_folder, replication_folder, device):
    content_file = path.join(source_folder, "praefcke__tuebingen_neckarfront.jpg")
    content_image = read_image(content_file).to(device)

    style_file = path.join(source_folder, "kandinsky__composition_vii.jpg")
    style_image = read_image(style_file).to(device)

    nst = GatysEckerBethge2015NST(impl_params=False).to(device)
    nst.content_loss.set_target(content_image)
    nst.style_loss.set_target(style_image)

    style_layers = nst.style_loss.layers
    layers_configs = [style_layers[: idx + 1] for idx in range(len(style_layers))]

    weight_ratios = (1e-5, 1e-4, 1e-3, 1e-2)

    for layers, weight_ratio in itertools.product(layers_configs, weight_ratios):
        row_label = layers[-1].replace("relu_", "Conv")
        column_label = "{:.0e}".format(weight_ratio)
        print(f"Replicating Figure 3 row {row_label} and column {column_label}")

        nst.style_loss.score_weight = nst.content_loss.score_weight / weight_ratio
        nst.style_loss.layers = layers
        nst.style_loss.layer_weights = [1.0 / len(layers)] * len(layers)

        utils.make_reproducible()
        input_image = utils.get_input_image("random", content_image, style_image.data)
        output_image = nst(input_image, NUM_STEPS, quiet=True)

        output_file = path.join(
            replication_folder, "fig_3__{}__{}.jpg".format(row_label, column_label)
        )
        print(f"Saving result to {output_file}")
        write_image(output_image, output_file)


if __name__ == "__main__":
    root = utils.get_pystiche_root(__file__)
    image_root = path.join(root, "images")
    device = None

    image_root = path.abspath(path.expanduser(image_root))
    source_folder = path.join(image_root, "source")
    replication_root = path.join(
        image_root, "replication", path.splitext(path.basename(__file__))[0]
    )
    device = utils.get_device(device)

    utils.print_replication_info(
        title="A Neural Algorithm of Artistic Style",
        url="https://arxiv.org/abs/1508.06576",
        author="Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge",
        year=2015,
    )
    replication_folder = path.join(replication_root, "paper")
    figure_2(source_folder, replication_folder, device)
    utils.print_sep_line()
    figure_3(source_folder, replication_folder, device)
    utils.print_sep_line()

    replication_folder = path.join(replication_root, "implementation")
    figure_2(source_folder, replication_folder, device, impl_params=True)
