from typing import Optional
from PIL import Image, ImageOps
from typing import Tuple
from pystiche.data import (
    NoLicense,
    PublicDomainLicense,
    CreativeCommonsLicense,
    DownloadableImage,
    DownloadableImageCollection,
)

__all__ = ["li_wand_2016_images"]


def image_note(url: str, mirror: bool = False) -> str:
    note = "The image is cropped"
    if mirror:
        note += " and mirrored"
    return f"{note}. The unprocessed image can be downloaded from {url}"


def transforms(image: str):
    def crop_transform(
        image: Image.Image, left_bottom_coord: Tuple[int, int], width: int, height: int,
    ) -> Image.Image:
        left = left_bottom_coord[0]
        right = left + width
        upper = left_bottom_coord[1]
        lower = upper + height
        box = (left, upper, right, lower)
        return image.crop(box)

    def resize_transform(image: Image.Image) -> Image.Image:
        width, height = image.size
        aspect_ratio = width / height
        new_height = 384
        new_width = round(new_height * aspect_ratio)
        new_size = (new_width, new_height)
        return image.resize(new_size, Image.BICUBIC)

    if image == "mricon":

        def image_transform(image: Image.Image) -> Image.Image:
            left_bottom_coord = (12, 30)
            width = 682
            height = 930
            return crop_transform(image, left_bottom_coord, width, height)

    elif image == "lydhode":

        def image_transform(image: Image.Image) -> Image.Image:
            image = ImageOps.mirror(image)
            left_bottom_coord = (462, 211)
            width = 1386
            height = 1843
            return crop_transform(image, left_bottom_coord, width, height)

    elif image == "theilr":

        def image_transform(image: Image.Image) -> Image.Image:
            left_bottom_coord = (486, 159)
            width = 1642
            height = 2157
            return crop_transform(image, left_bottom_coord, width, height)

    else:
        # TODO: add message
        raise RuntimeError

    def transform(image: Image.Image) -> Image.Image:
        return resize_transform(image_transform(image))

    return transform


def li_wand_2016_images(
    root: Optional[str] = None, download: bool = True, overwrite: bool = False
):

    images = {
        "emma": DownloadableImage(
            "https://live.staticflickr.com/1/2281680_656225393e_o_d.jpg",
            author="monsieuricon (mricon)",
            title="Emma",
            date="17.12.2004",
            transform=transforms("mricon"),
            license=CreativeCommonsLicense(("by", "sa"), "2.0"),
            note=image_note("https://www.flickr.com/photos/mricon/2281680/"),
            md5="7a10a2479864f394b4f06893b9202915",
        ),
        "jenny": DownloadableImage(
            "https://live.staticflickr.com/8626/16426686859_f882b3d317_o_d.jpg",
            author="Vidar Schiefloe (lydhode)",
            title="Jenny",
            date="06.02.2015",
            license=CreativeCommonsLicense(("by", "sa"), "2.0"),
            transform=transforms("lydhode"),
            note=image_note(
                "https://www.flickr.com/photos/lydhode/16426686859/", mirror=True,
            ),
            md5="5b3442909ff850551c9baea433319508",
        ),
        "blue_bottle": DownloadableImage(
            "https://raw.githubusercontent.com/chuanli11/CNNMRF/master/data/content/potrait1.jpg",
            title="Blue Bottle",
            author="Christopher Michel (cmichel67)",
            date="02.09.2014",
            license=NoLicense(),
            note=image_note("https://www.flickr.com/photos/cmichel67/15112861945"),
            md5="cb29d11ef6e1be7e074aa58700110e4f",
        ),
        "self-portrait": DownloadableImage(
            "https://raw.githubusercontent.com/chuanli11/CNNMRF/master/data/style/picasso.jpg",
            title="Self-Portrait",
            author="Pablo Ruiz Picasso",
            date="1907",
            license=PublicDomainLicense(1973),
            note=image_note("https://www.pablo-ruiz-picasso.net/images/works/57.jpg"),
            md5="4bd9c963fd52feaa940083f07e259aea",
        ),
        "s": DownloadableImage(
            "https://live.staticflickr.com/7409/9270411440_cdc2ee9c35_o_d.jpg",
            author="theilr",
            title="S",
            date="18.09.2011",
            license=CreativeCommonsLicense(("by", "sa"), "2.0"),
            transform=transforms("theilr"),
            note=image_note("https://www.flickr.com/photos/theilr/9270411440/"),
            md5="5d78432b5ca703bb85647274a5e41656",
        ),
        "composition_viii": DownloadableImage(
            "https://www.wassilykandinsky.net/images/works/50.jpg",
            title="Composition VIII",
            author="Wassily Kandinsky",
            date="1923",
            license=PublicDomainLicense(1944),
            md5="c39077aaa181fd40d7f2cd00c9c09619",
        ),
    }

    return DownloadableImageCollection(
        images, root=root, download=download, overwrite=overwrite
    )
