from os import path
import torch
import pystiche
from pystiche.image import read_image, write_image
from pystiche.image.transforms.functional import (
    resize,
    rgb_to_yuv,
    yuv_to_rgb,
    rgb_to_grayscale,
    grayscale_to_fakegrayscale,
    transform_channels_affinely,
)
from pystiche.papers import (
    GatysEtAl2017NSTPyramid,
    GatysEtAl2017SpatialControlNSTPyramid,
)
from pystiche.cuda import abort_if_cuda_memory_exausts
import utils


def display_replication_info(figure, impl_params):
    params = "implementation" if impl_params else "paper"
    print(f"Replicating {figure} with {params} parameters")


def display_saving_info(output_file):
    print(f"Saving result to {output_file}")


def perform_nst(content_image, style_image, impl_params, device):
    nst_pyramid = GatysEtAl2017NSTPyramid(impl_params).to(device)

    content_image = nst_pyramid.max_resize(content_image)
    style_image = nst_pyramid.max_resize(style_image)

    utils.make_reproducible()
    input_image = utils.get_input_image("content", content_image=content_image)

    nst = nst_pyramid.image_optimizer
    nst.content_loss.set_target(content_image)
    nst.style_loss.set_target(style_image)

    return nst_pyramid(input_image, quiet=True)[-1]


def figure_2(source_folder, guides_root, replication_folder, device, impl_params):
    def perform_guided_nst(
        content_image,
        content_guides,
        style_images,
        style_guides,
        guide_names,
        impl_params,
        device,
    ):
        nst_pyramid = GatysEtAl2017SpatialControlNSTPyramid(
            len(guide_names), impl_params, guide_names
        )
        nst_pyramid = nst_pyramid.to(device)

        content_image = nst_pyramid.max_resize(content_image)
        style_images = [nst_pyramid.max_resize(image) for image in style_images]

        content_guides = [
            nst_pyramid.max_resize(guide, binarize=True) for guide in content_guides
        ]
        style_guides = [
            nst_pyramid.max_resize(guide, binarize=True) for guide in style_guides
        ]

        utils.make_reproducible()
        input_image = utils.get_input_image("content", content_image=content_image)

        nst = nst_pyramid.image_optimizer
        nst.content_loss.set_target(content_image)
        for style_loss, content_guide, style_guide, style_image in zip(
            nst.style_losses, content_guides, style_guides, style_images
        ):
            style_loss.set_input_guide(content_guide)
            style_loss.set_target_guide(style_guide)
            style_loss.set_target(style_image)

        return nst_pyramid(input_image, quiet=True)[-1]

    @abort_if_cuda_memory_exausts
    def figure_2_d(content_image, style_image):
        display_replication_info("Figure 2 (d)", impl_params)
        output_image = perform_nst(content_image, style_image, impl_params, device)

        output_file = path.join(replication_folder, "fig_2__d.jpg")
        display_saving_info(output_file)
        write_image(output_image, output_file)

    @abort_if_cuda_memory_exausts
    def figure_2_ef(
        label,
        content_image,
        content_house_guide,
        content_sky_guide,
        style_house_image,
        style_house_guide,
        style_sky_image,
        style_sky_guide,
    ):
        guide_names = ("house", "sky")

        content_guides = (content_house_guide, content_sky_guide)
        style_images = (style_house_image, style_sky_image)
        style_guides = (style_house_guide, style_sky_guide)

        display_replication_info(f"Figure 2 ({label})", impl_params)
        output_image = perform_guided_nst(
            content_image,
            content_guides,
            style_images,
            style_guides,
            guide_names,
            impl_params,
            device,
        )

        output_file = path.join(replication_folder, "fig_2__{}.jpg".format(label))
        display_saving_info(output_file)
        write_image(output_image, output_file)

    content_file = path.join(source_folder, "house_concept_tillamook.jpg")
    content_image = read_image(content_file).to(device)
    content_guides = utils.read_guides(guides_root, content_file, device)

    style1_file = path.join(source_folder, "watertown.jpg")
    style1_image = read_image(style1_file).to(device)
    style1_guides = utils.read_guides(guides_root, style1_file, device)

    style2_file = path.join(source_folder, "van_gogh__wheat_field_with_cypresses.jpg")
    style2_image = read_image(style2_file).to(device)
    style2_guides = utils.read_guides(guides_root, style2_file, device)

    figure_2_d(content_image, style1_image)

    figure_2_ef(
        "e",
        content_image,
        content_guides["house"],
        content_guides["sky"],
        style1_image,
        style1_guides["house"],
        style1_image,
        style1_guides["sky"],
    )

    figure_2_ef(
        "f",
        content_image,
        content_guides["house"],
        content_guides["sky"],
        style1_image,
        style1_guides["house"],
        style2_image,
        style2_guides["sky"],
    )


def figure_3(source_folder, replication_folder, device, impl_params):
    def calculate_channelwise_mean_covariance(image):
        batch_size, num_channels, height, width = image.size()
        num_pixels = height * width
        image = image.view(batch_size, num_channels, num_pixels)

        mean = torch.mean(image, dim=2, keepdim=True)

        image_centered = image - mean
        cov = torch.bmm(image_centered, image_centered.transpose(1, 2)) / num_pixels

        return mean, cov

    def match_channelwise_statistics(input, target, method):
        input_mean, input_cov = calculate_channelwise_mean_covariance(input)
        target_mean, target_cov = calculate_channelwise_mean_covariance(target)

        input_cov, target_cov = [cov.squeeze(0) for cov in (input_cov, target_cov)]
        if method == "image_analogies":
            matrix = torch.mm(
                pystiche.msqrt(target_cov), torch.inverse(pystiche.msqrt(input_cov))
            )
        elif method == "cholesky":
            matrix = torch.mm(
                torch.cholesky(target_cov), torch.inverse(torch.cholesky(input_cov))
            )
        else:
            # FIXME: add error message
            raise ValueError
        matrix = matrix.unsqueeze(0)

        bias = target_mean - torch.bmm(matrix, input_mean)

        return transform_channels_affinely(input, matrix, bias)

    @abort_if_cuda_memory_exausts
    def figure_3_c(content_image, style_image):
        display_replication_info("Figure 3 (c)", impl_params)
        output_image = perform_nst(content_image, style_image, impl_params, device)

        output_file = path.join(replication_folder, "fig_3__c.jpg")
        display_saving_info(output_file)
        write_image(output_image, output_file)

    @abort_if_cuda_memory_exausts
    def figure_3_d(content_image, style_image):
        content_image_yuv = rgb_to_yuv(content_image)
        content_luminance = grayscale_to_fakegrayscale(content_image_yuv[:, 0])
        content_chromaticity = content_image_yuv[:, 1:]

        style_luminance = grayscale_to_fakegrayscale(rgb_to_grayscale(style_image))

        display_replication_info("Figure 3 (d)", impl_params)
        output_luminance = perform_nst(
            content_luminance, style_luminance, impl_params, device
        )
        output_luminance = torch.mean(output_luminance, dim=1, keepdim=True)
        output_chromaticity = resize(content_chromaticity, output_luminance.size()[2:])
        output_image_yuv = torch.cat((output_luminance, output_chromaticity), dim=1)
        output_image = yuv_to_rgb(output_image_yuv)

        output_file = path.join(replication_folder, "fig_3__d.jpg")
        display_saving_info(output_file)
        write_image(output_image, output_file)

    @abort_if_cuda_memory_exausts
    def figure_3_e(content_image, style_image, method="cholesky"):
        style_image = match_channelwise_statistics(style_image, content_image, method)

        display_replication_info("Figure 3 (e)", impl_params)
        output_image = perform_nst(content_image, style_image, impl_params, device)

        output_file = path.join(replication_folder, "fig_3__e.jpg")
        display_saving_info(output_file)
        write_image(output_image, output_file)

    content_file = path.join(
        source_folder, "janssen__schultenhof_mettingen_bauerngarten_8.jpg"
    )
    content_image = read_image(content_file).to(device)

    style_file = path.join(source_folder, "van_gogh__starry_night_over_rhone.jpg")
    style_image = read_image(style_file).to(device)

    figure_3_c(content_image, style_image)
    figure_3_d(content_image, style_image)
    figure_3_e(content_image, style_image)


if __name__ == "__main__":
    root = utils.get_pystiche_root(__file__)
    image_root = path.join(root, "images")
    device = None

    image_root = path.abspath(path.expanduser(image_root))
    source_folder = path.join(image_root, "source")
    guides_root = path.join(image_root, "guides")
    replication_root = path.join(
        image_root, "replication", path.splitext(path.basename(__file__))[0]
    )
    device = utils.get_device(device)

    utils.print_replication_info(
        title="Controlling Perceptual Factors in Neural Style Transfer",
        url="http://openaccess.thecvf.com/content_cvpr_2017/papers/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.pdf",
        author="Leon Gatys et. al.",
        year=2017,
    )
    for impl_params in (True, False):
        replication_folder = path.join(
            replication_root, "implementation" if impl_params else "paper"
        )

        figure_2(source_folder, guides_root, replication_folder, device, impl_params)
        utils.print_sep_line()
        figure_3(source_folder, replication_folder, device, impl_params)
        utils.print_sep_line()
