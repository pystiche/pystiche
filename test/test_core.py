import unittest
import torch
import pystiche


class Tester(unittest.TestCase):
    def test_LossDict_float(self):
        losses = [(str(i), torch.tensor(i)) for i in range(3)]
        loss_dict = pystiche.LossDict(losses)

        self.assertAlmostEqual(loss_dict.item(), 3.0)
        self.assertAlmostEqual(loss_dict.item(), float(loss_dict))


if __name__ == "__main__":
    unittest.main()
