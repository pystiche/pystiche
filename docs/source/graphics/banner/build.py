import torch

from pystiche.image import extract_image_size, read_image, write_image


def main(images, space_width_factor=0.1, annotation_width_factor=0.2):
    heights, widths = extract_image_sizes(images)
    pad_images(images, heights)

    space_width = int(space_width_factor * max(widths))
    banner, anchors = create_banner(images, heights, widths, space_width=space_width)

    annotation_width = int(annotation_width_factor * min(max(heights), max(widths)))
    banner = annotate_banner(banner, anchors, width=annotation_width)
    write_image(banner, "banner.jpg")


def extract_image_sizes(images):
    return zip(*[extract_image_size(image) for image in images])


def pad_images(images, heights):
    max_height = max(heights)
    padding_needed = any(height != max_height for height in heights)

    if not padding_needed:
        return images

    # FIXME: implement
    raise RuntimeError


def create_banner(images, heights, widths, space_width=0):
    height = max(heights)

    if space_width <= 0:
        banner = torch.cat(images, dim=3)
        anchors = [(height // 2, width) for width in cumsum(widths)[:-1]]
        return banner, anchors

    space = torch.ones(1, 3, height, space_width)

    banner = torch.cat(intersperse(images, space), dim=3)
    anchors = [
        (height // 2, width + int(space_width * (idx + 0.5)))
        for idx, width in enumerate(cumsum(widths)[:-1])
    ]

    return banner, anchors


def annotate_banner(banner, anchors, width=40):
    line_width = int(width * 0.1)

    def select_area(y_range, x_range, offset=(0, 0)):
        y_slice = slice(y_range[0] + offset[0], y_range[1] + 1 + offset[0])
        x_slice = slice(x_range[0] + offset[1], x_range[1] + 1 + offset[1])
        return slice(None), slice(None), y_slice, x_slice

    def add_plus(banner, anchor):
        # vertical line
        area = select_area((-width, width), (-line_width, line_width), offset=anchor)
        banner[area] = 0.0

        # horizontal line
        area = select_area((-line_width, line_width), (-width, width), offset=anchor)
        banner[area] = 0.0

        return banner

    def add_eq(banner, anchor):
        # top line
        offset = (anchor[0] + width // 3, anchor[1])
        area = select_area((-line_width, line_width), (-width, width), offset=offset)
        banner[area] = 0.0

        # top line
        offset = (anchor[0] - width // 3, anchor[1])
        area = select_area((-line_width, line_width), (-width, width), offset=offset)
        banner[area] = 0.0

        return banner

    for anchor in anchors[:-1]:
        banner = add_plus(banner, anchor)

    return add_eq(banner, anchors[-1])


def read_images():
    content_image = read_image("content.jpg")
    style_image = read_image("style.jpg")
    stylized_image = read_image("stylized.jpg")

    return content_image, style_image, stylized_image


def cumsum(seq):
    total = 0
    res = []
    for item in seq:
        total += item
        res.append(total)
    return res


# Copied from https://stackoverflow.com/a/6300649/1654607
def intersperse(seq, item):
    res = [item] * (2 * len(seq) - 1)
    res[::2] = seq
    return res


if __name__ == "__main__":
    images = read_images()
    main(images)
