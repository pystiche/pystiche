import logging
import sys

from pystiche.data import (
    DownloadableImage,
    DownloadableImageCollection,
    PixabayLicense,
    PublicDomainLicense,
)
from pystiche.optim import OptimLogger

__all__ = ["demo_images", "demo_logger"]


def demo_images() -> DownloadableImageCollection:
    return DownloadableImageCollection(
        {
            "dancing": DownloadableImage(
                "https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg",
                md5="0a2df538901452d639170a2ed89815a4",
            ),
            "picasso": DownloadableImage(
                "https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg",
                md5="d1d60fc3f9d0b22d2d826c47934a37ea",
            ),
            "bird1": DownloadableImage(
                "https://cdn.pixabay.com/photo/2016/01/14/11/26/bird-1139734_960_720.jpg",
                file="bird1.jpg",
                author="gholmz0",
                date="09.03.2013",
                license=PixabayLicense(),
                md5="36e5fef725943a5d1d22b5048095da86",
            ),
            "paint": DownloadableImage(
                "https://cdn.pixabay.com/photo/2017/07/03/20/17/abstract-2468874_960_720.jpg",
                file="paint.jpg",
                author="garageband",
                date="03.07.2017",
                license=PixabayLicense(),
                md5="a991e222806ef49d34b172a67cf97d91",
            ),
            "bird2": DownloadableImage(
                "https://cdn.pixabay.com/photo/2013/03/12/17/53/bird-92956_960_720.jpg",
                file="bird2.jpg",
                author="12019",
                date="09.04.2012",
                license=PixabayLicense(),
                md5="8c5b608bd579d931e2cfe7229840fe9b",
            ),
            "mosaic": DownloadableImage(
                "https://upload.wikimedia.org/wikipedia/commons/2/23/Mosaic_ducks_Massimo.jpg",
                file="mosaic.jpg",
                author="Marie-Lan Nguyen",
                date="2006",
                license=PublicDomainLicense(),
                md5="5b60cd1724395f7a0c21dc6dd006f8ae",
            ),
            "lemgo_castle": DownloadableImage(
                "https://free-images.com/lg/7a43/lemgo_brake_castle_bridge.jpg",
                file="lemgo_castle.jpg",
                author="Christian Hänsel",
                date="23.03.2014",
                md5="9db982e6c5cd17d5c0a9e93171d4df29",
                guides=DownloadableImageCollection(
                    {
                        "building": DownloadableImage(
                            "building.png",  # FIXME
                            md5="35843acf145f4402d01bb2e1911def7e",
                        ),
                        "sky": DownloadableImage(
                            "sky.png", md5="c0c956429c52069302c29a3737d6d211",  # FIXME
                        ),
                    },
                ),
            ),
            "segovia_fortress": DownloadableImage(
                "https://images.unsplash.com/photo-1579541814924-49fef17c5be5?ixlib=rb-1.2.1&q=80&fm=jpg&crop=entropy&cs=tinysrgb&dl=birmingham-museums-trust-sJr8LDyEf7k-unsplash.jpg",
                file="segovia_fortress.jpg",
                title="Alcazar Segovia",
                author="David Roberts",
                date="1836",
                license=PublicDomainLicense(1864),
                md5="fb66841d673afed12e849641f0723acf",
                guides=DownloadableImageCollection(
                    {
                        "building": DownloadableImage(
                            "building.png",  # FIXME
                            md5="11fd272d7439539bbcfd3e765497a4a6",
                        ),
                        "sky": DownloadableImage(
                            "sky.png", md5="4a216b582d9ab10a8919344e2596cf6b",  # FIXME
                        ),
                        "surroundings": DownloadableImage(
                            "surroundings.png",  # FIXME
                            md5="bbc45cc9126a68c89c7408eccbf973c3",
                        ),
                    }
                ),
            ),
        }
    )


def demo_logger() -> OptimLogger:
    logger = logging.getLogger("demo_logger")
    logger.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    return OptimLogger(logger)
