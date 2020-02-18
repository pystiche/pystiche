from typing import Optional
from pystiche.image.data import (
    PublicDomainLicense,
    CreativeCommonsLicense,
    DownloadableImage,
    DownloadableImageCollection,
)

__all__ = ["gatys_et_al_2017_images"]


def gatys_et_al_2017_images(
    root: Optional[str] = None, download: bool = True, force: bool = False
):
    images = {
        "house": DownloadableImage(
            "https://associateddesigns.com/sites/default/files/plan_images/main/craftsman_house_plan_tillamook_30-519-picart.jpg",
            title="House Concept Tillamook",
            date="2014",
            md5="5629bf7b24a7c98db2580ec2a8d784e9",
        ),
        "watertown": DownloadableImage(
            "https://ae01.alicdn.com/img/pb/136/085/095/1095085136_084.jpg",
            title="Watertown",
            author="Shop602835 Store",
            md5="4cc98a503da5ce6eab0649b09fd3cf77",
        ),
        "wheat_field": DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Wheat-Field-with-Cypresses-%281889%29-Vincent-van-Gogh-Met.jpg/1920px-Wheat-Field-with-Cypresses-%281889%29-Vincent-van-Gogh-Met.jpg",
            title="Wheat Field with Cypresses",
            author="Vincent van Gogh",
            date="1889",
            license=PublicDomainLicense(1890),
            md5="bfd085d7e800459c8ffb44fa404f73c3",
        ),
        "schultenhof": DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/8/82/Schultenhof_Mettingen_Bauerngarten_8.jpg",
            title="Schultenhof Mettingen Bauerngarten 8",
            author="J.-H. Janßen",
            date="July 2010",
            license=CreativeCommonsLicense(("by", "sa"), "3.0"),
            md5="23f75f148b7b94d932e599bf0c5e4c8e",
        ),
        "starry_night": DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/9/94/Starry_Night_Over_the_Rhone.jpg",
            title="Starry Night Over the Rhone",
            author="Vincent Willem van Gogh",
            date="1888",
            license=PublicDomainLicense(1890),
            md5="406681ec165fa55c26cb6f988907fe11",
        ),
    }
    return DownloadableImageCollection(
        images, root=root, download=download, force=force
    )
