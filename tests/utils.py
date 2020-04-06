import contextlib
from os import path
import tempfile
import shutil
import unittest
import pyimagetest
import numpy as np
import torch
from PIL import Image

__all__ = ["PysticheTestCase", "get_tmp_dir", "skip_if_cuda_not_available"]


class PysticheImageBackend(pyimagetest.ImageBackend):
    @property
    def native_image_type(self):
        return torch.Tensor

    def import_image(self, file):
        pil_image = Image.open(file)
        np_image = np.array(pil_image, dtype=np.float32) / 255.0
        pystiche_image = torch.from_numpy(np_image).permute((2, 0, 1)).unsqueeze(0)
        return pystiche_image

    def export_image(self, image):
        image = image.detach().cpu()
        if image.dim() == 4 and image.size()[0] == 1:
            image = image.squeeze(0)
        return image.permute((1, 2, 0)).numpy()


class PysticheTestCase(pyimagetest.ImageTestCase):
    project_root = path.abspath(path.join(path.dirname(__file__), ".."))
    package_name = "pystiche"

    @property
    def package_root(self):
        return path.join(self.project_root, self.package_name)

    @property
    def test_root(self):
        return path.join(self.project_root, "tests")

    @property
    def test_assets_root(self):
        return path.join(self.test_root, "assets")

    def setUp(self):
        # the builtin torchvision backend is removed, since it operates on the same
        # native image type (torch.Tensor), which renders the automatic inference of
        # the image backend impossible
        self.remove_image_backend("torchvision")
        self.add_image_backend(self.package_name, PysticheImageBackend())

    def default_image_backend(self) -> str:
        return self.package_name

    def default_image_file(self) -> str:
        return path.join(self.test_assets_root, "image", "test_image.png")

    def load_batched_image(self, batch_size=1, file=None):
        return self.load_image(file=file).repeat(batch_size, 1, 1, 1)

    def load_single_image(self, file=None):
        return self.load_batched_image(file=file).squeeze(0)

    def assertFloatAlmostEqual(self, actual, desired, **kwargs):
        if isinstance(actual, torch.Tensor):
            actual = actual.item()

        if isinstance(desired, torch.Tensor):
            desired = desired.item()

        np.testing.assert_almost_equal(actual, desired, **kwargs)

    def assertTensorAlmostEqual(
        self, actual, desired, check_dtype=True, check_device=True, **kwargs
    ):
        def cast(x):
            return x.detach().cpu().numpy()

        if check_dtype:
            self.assertEqual(actual.dtype, desired.dtype)

        if check_device:
            self.assertEqual(actual.device, desired.device)

        np.testing.assert_allclose(cast(actual), cast(desired), **kwargs)

    def assertNamedChildrenEqual(self, actuals, desireds, equality_sufficient=False):
        self.assertCountEqual(actuals, desireds)
        for actual, desired in zip(actuals, desireds):
            actual_name, actual_module = actual
            desired_name, desired_module = actual
            self.assertEqual(actual_name, desired_name)
            if equality_sufficient:
                self.assertEqual(actual_module, desired_module)
            else:
                self.assertIs(actual_module, desired_module)

    def assertTensorDictAlmostEqual(self, actual, desired, **kwargs):
        self.assertCountEqual(actual, desired)
        self.assertEqual(set(actual.keys()), set(desired.keys()))
        for key in actual.keys():
            self.assertTensorAlmostEqual(actual[key], desired[key], **kwargs)


@contextlib.contextmanager
def get_tmp_dir(**mkdtemp_kwargs):
    tmp_dir = tempfile.mkdtemp(**mkdtemp_kwargs)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


skip_if_cuda_not_available = unittest.skipIf(
    not torch.cuda.is_available(), "CUDA is not available."
)
