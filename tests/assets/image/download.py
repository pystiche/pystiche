from os import path
from urllib.request import urlretrieve


def main(filename="test_image"):
    # This image is cleared for unrestricted usage. For details see
    # http://www.r0k.us/graphics/kodak/
    url = "http://www.r0k.us/graphics/kodak/kodak/kodim15.png"

    here = path.abspath(path.dirname(__file__))
    file = path.join(here, f"{filename}{path.splitext(url)[1]}")

    urlretrieve(url, filename=file)


if __name__ == "__main__":
    main()
