from argparse import Namespace
from os import path

import dill

__all__ = ["store_asset"]


def store_asset(input, params, output, file, ext=".asset", verify_loadability=True):
    asset = Namespace(
        input=Namespace(**input), params=Namespace(**params), output=Namespace(**output)
    )
    file = path.splitext(file)[0] + ext

    with open(file, "wb") as fh:
        dill.dump(asset, fh)

    if verify_loadability:
        with open(file, "rb") as fh:
            dill.load(fh)
