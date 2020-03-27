import unittest
import torch
from pystiche.misc import cuda
from utils import skip_if_cuda_not_available


class TestCase(unittest.TestCase):
    @staticmethod
    def create_large_cuda_tensor(size_in_gb=256):
        return torch.empty((size_in_gb, *[1024] * 3), device="cuda", dtype=torch.uint8)

    @skip_if_cuda_not_available
    def test_use_cuda_out_of_memory_error(self):
        with self.assertRaises(cuda.CudaOutOfMemoryError):
            with cuda.use_cuda_out_of_memory_error():
                self.create_large_cuda_tensor()

    @skip_if_cuda_not_available
    def test_abort_if_cuda_memory_exausts(self):
        @cuda.abort_if_cuda_memory_exausts
        def fn():
            self.create_large_cuda_tensor()

        with self.assertWarns(cuda.CudaOutOfMemoryWarning):
            fn()
