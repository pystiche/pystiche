import os

__all__ = ["home"]


def home() -> str:
    return os.getenv(
        "PYSTICHE_HOME", os.path.expanduser(os.path.join("~", ".cache", "pystiche"))
    )
