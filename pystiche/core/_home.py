import os

__all__ = ["home"]


def home():
    root = os.getenv(
        "PYSTICHE_HOME", os.path.expanduser(os.path.join("~", ".cache", "pystiche"))
    )
    os.makedirs(root, exist_ok=True)
    return root
