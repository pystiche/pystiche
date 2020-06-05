import os

__all__ = ["home"]


def home() -> str:
    r"""Local directory to save downloaded images and guides. Defaults to
    ``~/.cache/pystiche`` but can be overwritten with the ``PYSTICHE_HOME`` environment
    variable.
    """
    return os.getenv(
        "PYSTICHE_HOME", os.path.expanduser(os.path.join("~", ".cache", "pystiche"))
    )
