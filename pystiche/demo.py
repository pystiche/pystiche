import logging
import sys
import warnings
from datetime import datetime

from pystiche.data import (
    DownloadableImage,
    DownloadableImageCollection,
    ExpiredCopyrightLicense,
    PixabayLicense,
    PublicDomainLicense,
)
from pystiche.optim import OptimLogger

from .misc import build_deprecation_message, suppress_warnings

__all__ = ["images", "logger"]


def images() -> DownloadableImageCollection:
    """Collection of images used in the usage examples.

    .. note::

        You can use

        .. code-block:: python

            print(images())

        to get a list of all available images.
    """
    return DownloadableImageCollection(
        {
            "bird1": DownloadableImage(
                "https://download.pystiche.org/images/bird1.jpg",
                file="bird1.jpg",
                author="gholmz0",
                date="09.03.2013",
                license=PixabayLicense(),
                md5="ca869559701156c66fc1519ffc26123c",
                note="https://pixabay.com/photos/bird-wildlife-australian-bird-1139734/",
            ),
            "paint": DownloadableImage(
                "https://download.pystiche.org/images/paint.jpg",
                file="paint.jpg",
                author="garageband",
                date="03.07.2017",
                license=PixabayLicense(),
                md5="a991e222806ef49d34b172a67cf97d91",
                note="https://pixabay.com/de/photos/abstrakt-kunst-hintergrund-farbe-2468874/",
            ),
            "bird2": DownloadableImage(
                "https://download.pystiche.org/images/bird2.jpg",
                file="bird2.jpg",
                author="12019",
                date="09.04.2012",
                license=PixabayLicense(),
                md5="2398e8f6d102b9b8de7c74d4c73b71f4",
                note="https://pixabay.com/photos/bird-wildlife-sky-clouds-nature-92956/",
            ),
            "mosaic": DownloadableImage(
                "https://upload.wikimedia.org/wikipedia/commons/2/23/Mosaic_ducks_Massimo.jpg",
                file="mosaic.jpg",
                author="Marie-Lan Nguyen",
                date="2006",
                license=PublicDomainLicense(),
                md5="5b60cd1724395f7a0c21dc6dd006f8ae",
            ),
            "castle": DownloadableImage(
                "https://download.pystiche.org/images/castle.jpg",
                file="castle.jpg",
                author="Christian Hänsel",
                date="23.03.2014",
                license=PixabayLicense(),
                md5="9d085fac77065e0da66629162f759f90",
                note="https://pixabay.com/de/photos/lemgo-brake-schloss-br%C3%BCcke-310586/",
                guides=DownloadableImageCollection(
                    {
                        "building": DownloadableImage(
                            "https://download.pystiche.org/images/castle/building.png",
                            file="building.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="680bd875f4fdb919d446d6210cbe1035",
                        ),
                        "sky": DownloadableImage(
                            "https://download.pystiche.org/images/castle/sky.png",
                            file="sky.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="c4d5f18b3ca22a1ff879f40ca797ed26",
                        ),
                        "water": DownloadableImage(
                            "https://download.pystiche.org/images/castle/water.png",
                            file="water.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="2642552e8c98b6a63f24dd67bb54e2a3",
                        ),
                    },
                ),
            ),
            "church": DownloadableImage(
                "https://upload.wikimedia.org/wikipedia/commons/a/ad/Vincent_van_Gogh_-_The_Church_in_Auvers-sur-Oise%2C_View_from_the_Chevet_-_Google_Art_Project.jpg",
                file="church.jpg",
                author="Vincent van Gogh",
                title="The Church at Auvers",
                date="1890",
                license=ExpiredCopyrightLicense(1890),
                md5="fd866289498367afd72a5c9cf626073a",
                guides=DownloadableImageCollection(
                    {
                        "building": DownloadableImage(
                            "https://download.pystiche.org/images/church/building.png",
                            file="building.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="41256f62d2bd1560ed1e66623c8d9c9f",
                        ),
                        "sky": DownloadableImage(
                            "https://download.pystiche.org/images/church/sky.png",
                            file="sky.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="c87c7234d2788a1d2a4e1633723c794b",
                        ),
                        "surroundings": DownloadableImage(
                            "https://download.pystiche.org/images/church/surroundings.png",
                            file="surroundings.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="807a40beed727af81333edbd8eb89aff",
                        ),
                    }
                ),
            ),
            "cliff": DownloadableImage(
                "https://upload.wikimedia.org/wikipedia/commons/a/a4/Claude_Monet_-_Cliff_Walk_at_Pourville_-_Google_Art_Project.jpg",
                file="cliff.jpg",
                author="Claude Monet",
                title="The Cliff Walk at Pourville",
                date="1882",
                license=ExpiredCopyrightLicense(1926),
                md5="8f6b8101b484f17cea92a12ad27be31d",
                guides=DownloadableImageCollection(
                    {
                        "landscape": DownloadableImage(
                            "https://download.pystiche.org/images/cliff/landscape.png",
                            file="landscape.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="4d02ee8aaa610ced3ef12a5c82f29b81",
                        ),
                        "sky": DownloadableImage(
                            "https://download.pystiche.org/images/cliff/sky.png",
                            file="sky.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="8f76b7ed88ff1b23feaab1de51501ae1",
                        ),
                        "water": DownloadableImage(
                            "https://download.pystiche.org/images/cliff/water.png",
                            file="water.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="f0cab02afadf38d3262ce1c61a36f6da",
                        ),
                    }
                ),
            ),
        }
    )


def demo_images() -> DownloadableImageCollection:
    warnings.warn(
        build_deprecation_message(
            "The function demo_images", "0.6.0", info="It was renamed to images."
        )
    )
    return images()


def logger() -> OptimLogger:
    """Simple logger used in the usage examples.
    """
    msg = build_deprecation_message(
        "The function logger",
        "0.7.0",
        "See https://github.com/pmeier/pystiche/issues/434 for details.",
    )
    warnings.warn(msg, UserWarning)
    logger = logging.getLogger(
        name=f"pystiche_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    )
    logger.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    with suppress_warnings():
        return OptimLogger(logger)


def demo_logger() -> OptimLogger:
    warnings.warn(
        build_deprecation_message(
            "The function demo_logger", "0.6.0", info="It was renamed to logger."
        )
    )
    return logger()
