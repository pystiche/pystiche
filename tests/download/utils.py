import time
from datetime import datetime
from os import path
from time import sleep
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from torchvision.datasets.utils import calculate_md5

from tests.utils import get_tmp_dir

__all__ = [
    "rate_limited_urlopen",
    "retry",
    "assert_is_downloadable",
    "assert_downloads_correctly",
]

USER_AGENT = "pystiche/test_suite"


def limit_requests_per_time(min_secs_between_requests=2.0):
    last_requests = {}

    def outer_wrapper(fn):
        def inner_wrapper(request, *args, **kwargs):
            url = request.full_url if isinstance(request, Request) else request

            netloc = urlparse(url).netloc
            last_request = last_requests.get(netloc)
            if last_request is not None:
                now = datetime.now()

                elapsed_secs = (now - last_request).total_seconds()
                delta = min_secs_between_requests - elapsed_secs
                if delta > 0:
                    time.sleep(delta)

            try:
                return fn(request, *args, **kwargs)
            finally:
                last_requests[netloc] = datetime.now()

        return inner_wrapper

    return outer_wrapper


rate_limited_urlopen = limit_requests_per_time()(urlopen)


def retry(fn, times=1, wait=5.0):
    if not times:
        return fn()

    msgs = []
    for _ in range(times + 1):
        try:
            return fn()
        except AssertionError as error:
            msgs.append(str(error))
            sleep(wait)
    else:
        head = (
            f"Assertion failed {times + 1} times with {wait:.1f} seconds intermediate "
            f"wait time.\n"
        )
        raise AssertionError(
            "\n".join((head, *(f"{idx}: {error}" for idx, error in enumerate(msgs, 1))))
        )


def assert_response_ok(response, url=None):
    msg = f"The server returned status code {response.code}"
    if url is not None:
        msg += f" for the URL {url}"
    assert response.code == 200, msg


def assert_is_downloadable(url, times=1, wait=5.0):
    response = rate_limited_urlopen(
        Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    )
    retry(lambda: assert_response_ok(response, url), times=times - 1, wait=wait)


def default_downloader(url, root):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    file = path.join(root, path.basename(url))
    with rate_limited_urlopen(request) as response, open(file, "wb") as fh:
        assert_response_ok(response, url)
        fh.write(response.read())
    return file


def assert_downloads_correctly(
    url, md5, downloader=default_downloader, times=1, wait=5.0
):
    with get_tmp_dir() as root:
        file = retry(lambda: downloader(url, root), times=times - 1, wait=wait)
        assert calculate_md5(file) == md5, "The MD5 checksums mismatch"
