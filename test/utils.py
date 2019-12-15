from typing import Type
import pyimagetest
from os import path
import numpy as np
import torch
from PIL import Image
from pystiche.image.transforms import ImportFromPIL


class PysticheImageBackend(pyimagetest.ImageBackend):
    @property
    def native_image_type(self) -> Type[torch.FloatTensor]:
        return torch.FloatTensor

    def import_image(self, file: str) -> torch.FloatTensor:
        image = Image.open(file)
        transform = ImportFromPIL()
        return transform(image)

    def export_image(self, image: torch.FloatTensor) -> np.ndarray:
        return image.detach().cpu().squeeze(0).permute((1, 2, 0)).numpy()


class PysticheImageTestscae(pyimagetest.ImageTestcase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # the builtin torchvision backend is removed, since it operates on the same
        # native image type (torch.FloatTensor), which renders the automatic inference
        # of the image backend impossible
        self.remove_image_backend("torchvision")
        self.add_image_backend("pystiche", PysticheImageBackend())

    @property
    def default_test_image_file(self) -> str:
        # The test image was downloaded from
        # http://www.r0k.us/graphics/kodak/kodim15.html
        # and is cleared for unrestricted usage
        return path.join(path.dirname(__file__), "test_image.png")
