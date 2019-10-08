from os import path
from pystiche.image import read_image, write_image
from pystiche.papers import LiWand2016NSTPyramid
from pystiche.cuda import abort_if_cuda_memory_exausts
import utils


def perform_nst(content_image, style_image, impl_params, device):
    nst_pyramid = LiWand2016NSTPyramid(impl_params).to(device)

    content_image = nst_pyramid.max_resize(content_image)
    style_image = nst_pyramid.max_resize(style_image)

    utils.make_reproducible()
    starting_point = "content" if impl_params else "random"
    input_image = utils.get_input_image(starting_point, content_image=content_image)

    nst = nst_pyramid.image_optimizer
    nst.content_loss.set_target(content_image)
    nst.style_loss.set_target(style_image)

    return nst_pyramid(input_image, quiet=True)[-1]


@abort_if_cuda_memory_exausts
def figure_6(source_folder, replication_folder, device, impl_params):
    content_files = ("jeffrey_dennard.jpg", "theilr__s.jpg")
    style_files = ("picasso__self-portrait_1907.jpg", "kandinsky__composition_viii.jpg")
    locations = ("top", "bottom")

    for content_file, style_file, location in zip(
        content_files, style_files, locations
    ):
        content_image = read_image(path.join(source_folder, content_file)).to(device)
        style_image = read_image(path.join(source_folder, style_file)).to(device)

        params = "implementation" if impl_params else "paper"
        print(f"Replicating the {location} half of figure 6 with {params} parameters")
        output_image = perform_nst(content_image, style_image, impl_params, device)

        output_file = path.join(replication_folder, "fig_6__{}.jpg".format(location))
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
        title="Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis",
        url="https://ieeexplore.ieee.org/document/7780641",
        author="Chuan Li and Michael Wand",
        year=2016,
    )
    for impl_params in (True, False):
        replication_folder = path.join(
            replication_root, "implementation" if impl_params else "paper"
        )

        figure_6(source_folder, replication_folder, device, impl_params)
        utils.print_sep_line()
