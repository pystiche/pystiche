from os import path
from urllib.request import urlretrieve


def main(root, filename="test_image"):
    # This image is cleared for unrestricted usage. For details see
    # http://www.r0k.us/graphics/kodak/
    url = "http://www.r0k.us/graphics/kodak/kodak/kodim15.png"

    filename = path.join(root, f"{filename}{path.splitext(url)[1]}")
    urlretrieve(url, filename=filename)


if __name__ == "__main__":
    root = path.join(path.dirname(__file__), "image")
    main(root)
