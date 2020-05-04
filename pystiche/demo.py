from pystiche.data import (
    DownloadableImage,
    DownloadableImageCollection,
    PixabayLicense,
    PublicDomainLicense,
)

__all__ = ["demo_images"]


def demo_images():
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
                md5="d42444d3cd0afa47f07066cd083d6cea",
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
                md5="dda3e1d0f93f783de823b4f91129d44e",
            ),
            "mosaic": DownloadableImage(
                "https://upload.wikimedia.org/wikipedia/commons/2/23/Mosaic_ducks_Massimo.jpg",
                file="mosaic.jpg",
                author="Marie-Lan Nguyen",
                date="2006",
                license=PublicDomainLicense(),
                md5="5b60cd1724395f7a0c21dc6dd006f8ae",
            ),
        }
    )
