from pystiche.data import DownloadableImage, DownloadableImageCollection

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
        }
    )
