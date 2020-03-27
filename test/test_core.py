import torch
import pystiche
from utils import PysticheTestCase


class TestCase(PysticheTestCase):
    def test_LossDict_float(self):
        losses = [(str(i), torch.tensor(i)) for i in range(3)]
        loss_dict = pystiche.LossDict(losses)

        self.assertAlmostEqual(loss_dict.item(), 3.0)
        self.assertAlmostEqual(loss_dict.item(), float(loss_dict))
