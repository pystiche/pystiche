import numpy as np
import pyimagetest
import pytest
from PIL import Image

import torch


class PysticheImageBackend(pyimagetest.ImageBackend):
    @property
    def native_image_type(self):
        return torch.Tensor

    def import_image(self, file):
        pil_image = Image.open(file)
        np_image = np.array(pil_image, dtype=np.float32) / 255.0

        pystiche_image = torch.from_numpy(np_image)
        if pystiche_image.dim() == 2:
            pystiche_image = pystiche_image.unsqueeze(0)
        elif pystiche_image.dim() == 3:
            pystiche_image = pystiche_image.permute((2, 0, 1))
        pystiche_image = pystiche_image.unsqueeze(0)

        return pystiche_image

    def export_image(self, image):
        image = image.detach().cpu()
        if image.dim() == 4 and image.size()[0] == 1:
            image = image.squeeze(0)
        return image.permute((1, 2, 0)).numpy()


@pytest.fixture(scope="session", autouse=True)
def add_pystiche_image_backend():
    pyimagetest.remove_image_backend("torchvision")
    pyimagetest.add_image_backend("pystiche", PysticheImageBackend())
