from collections import OrderedDict
from urllib.parse import urljoin
from ..license import UnknownLicense
from .image import DownloadableImage
from .collection import DownloadableImageCollection

__all__ = ["NPRgeneral"]


def name_to_url(name: str) -> str:
    base = "http://gigl.scs.carleton.ca/sites/default/files/"
    if name == "rimlighting":
        name = "rim"
    elif name == "tomatoes":
        name = "tomato"
    return urljoin(base, f"{name}1024.jpg")


class NPRgeneralLicense(UnknownLicense):
    def __str__(self) -> str:
        return (
            "Although the authors of the NPRgeneral dataset, David Mould and "
            "Paul Rosin, claim that the license of  this images "
            "'permits distribution of modified versions', the actual license is "
            "unknown. Proceed to work with this image at your own risk."
        )


class NPRgeneral(DownloadableImageCollection):
    def __init__(self):
        image_meta = (
            ("angel", "Eole Wind", "41b058edd2091eec467d37054a8c01fb"),
            ("arch", "James Marvin Phelps", "9378e3a129f021548ba26b99ae445ec9"),
            ("athletes", "Nathan Congleton", "6b742bd1b46bf9882cc176e0d255adab"),
            ("barn", "MrClean1982", "32abf24dd439940cf9a1265965b5e910"),
            ("berries", "HelmutZen", "58fea65145fa0bd222800d4f8946d815"),
            ("cabbage", "Leonard Chien", "19304a80d2ca2389153562700f0aab53"),
            ("cat", "Theen Moy", "de3331050c660e688463fa86c76976c4"),
            ("city", "Rob Schneider", "84e17b078ede986ea3e8f70e0a24195e"),
            ("daisy", "mgaloseau", "f3901d987490613238ef01977f9fac77"),
            ("darkwoods", "JB Banks", "f4ccbf37b3d5d3a1ca734bb65464151b"),
            ("desert", "Charles Roffey", "4a9db691d203dd693b14090b9e49f791"),
            ("headlight", "Photos By Clark", "1d4723535ea7dee84969f0c082445ad5"),
            ("mac", "Martin Kenney", "4df4e0dafe5468b86138839f68beb84d"),
            ("mountains", "Jenny Pansing", "1ea43310f2d85b271827634c733b6257"),
            ("oparara", "trevorklatko", "1adcc54d9a390b2492ee6e71acac13fd"),
            ("rimlighting", "Paul Stevenson", "8878a60ccd81e7b2ff05089cdf54773b"),
            ("snow", "John Anes", "60d55728e28d39114d0035c04c9e4495"),
            ("tomatoes", "Greg Myers", "3ffcdc427060171aabf433181cef7c52"),
            ("toque", "sicknotepix", "34fcf032b5d87877a7fc95629efb94e8"),
            ("yemeni", "Richard Messenger", "dd1e81d885cdbdd44f8d504e3951fb48"),
        )
        images = OrderedDict(
            [
                (
                    name,
                    DownloadableImage(
                        name_to_url(name),
                        title=name,
                        author=author,
                        license=NPRgeneralLicense(),
                        md5=md5,
                    ),
                )
                for name, author, md5 in image_meta
            ]
        )
        super().__init__(images)
