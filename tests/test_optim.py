from datetime import datetime, timedelta
import torch
from pystiche import optim
from utils import PysticheTestCase


class TestLog(PysticheTestCase):
    pass


class TestMeter(PysticheTestCase):
    class TestFloatMeter(optim.FloatMeter):
        def __init__(self, window_size=10):
            super().__init__("float_meter", window_size=window_size)

        def __str__(self):
            pass

    def test_FloatMeter_count(self):
        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter()
        meter.update(vals.tolist())

        actual = meter.count
        desired = len(vals)
        self.assertEqual(actual, desired)

    def test_FloatMeter_last_val(self):
        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter()
        meter.update(vals.tolist())

        actual = meter.last_val
        desired = vals[-1].item()
        self.assertFloatAlmostEqual(actual, desired)

    def test_FloatMeter_global_sum(self):
        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter()
        meter.update(vals.tolist())

        actual = meter.global_sum
        desired = torch.sum(vals).item()
        self.assertFloatAlmostEqual(actual, desired, decimal=6)

    def test_FloatMeter_global_min(self):
        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter()
        meter.update(vals.tolist())

        actual = meter.global_min
        desired = torch.min(vals).item()
        self.assertFloatAlmostEqual(actual, desired)

    def test_FloatMeter_global_max(self):
        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter()
        meter.update(vals.tolist())

        actual = meter.global_max
        desired = torch.max(vals).item()
        self.assertFloatAlmostEqual(actual, desired)

    def test_FloatMeter_global_avg(self):
        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter()
        meter.update(vals.tolist())

        actual = meter.global_avg
        desired = torch.mean(vals).item()
        self.assertFloatAlmostEqual(actual, desired, decimal=6)

    def test_FloatMeter_global_avg_empty(self):
        meter = self.TestFloatMeter()
        with self.assertRaises(RuntimeError):
            meter.global_avg

    def test_FloatMeter_local_avg(self):
        window_size = 5

        torch.manual_seed(0)
        vals = torch.rand(10)

        meter = self.TestFloatMeter(window_size=window_size)
        meter.update(vals.tolist())

        actual = meter.local_avg
        desired = torch.mean(vals[-window_size:]).item()
        self.assertFloatAlmostEqual(actual, desired, decimal=6)

    def test_FloatMeter_local_avg_empty(self):
        meter = self.TestFloatMeter()
        with self.assertRaises(RuntimeError):
            meter.local_avg

    def test_AverageMeter_update_tensor(self):
        actual_meter = optim.AverageMeter("actual_meter")
        desired_meter = optim.AverageMeter("desired_meter")

        torch.manual_seed(0)
        vals = torch.rand(10)

        actual_meter.update(vals)
        desired_meter.update(vals.tolist())

        for attr in (
            "count",
            "last_val",
            "global_sum",
            "global_min",
            "global_max",
            "global_avg",
            "local_avg",
        ):
            actual = getattr(actual_meter, attr)
            desired = getattr(desired_meter, attr)
            self.assertAlmostEqual(actual, desired)

    def test_AverageMeter_str_smoke(self):
        for show_local_avg in (True, False):
            meter = optim.AverageMeter(
                "test_average_meter", show_local_avg=show_local_avg
            )
            self.assertIsInstance(str(meter), str)
            meter.update(0.0)
            self.assertIsInstance(str(meter), str)

    def test_ETAMeter(self):
        first_time = 1.0
        second_time = 3.0
        meter = optim.ETAMeter(1)

        meter.update(first_time)

        actual = meter.count
        desired = 0
        self.assertEqual(actual, desired)

        actual = meter.last_time
        desired = first_time
        self.assertEqual(actual, desired)

        meter.update(second_time)

        actual = meter.count
        desired = 1
        self.assertEqual(actual, desired)

        actual = meter.last_time
        desired = second_time
        self.assertEqual(actual, desired)

        actual = meter.last_val
        desired = second_time - first_time
        self.assertEqual(actual, desired)

    def test_ETAMeter_calculate_eta(self):
        meter = optim.ETAMeter(1)
        time_diff = 10.0

        actual = meter.calculate_eta(time_diff)
        desired = datetime.now() + timedelta(seconds=time_diff)
        self.assertAlmostEqual(actual, desired, delta=timedelta(seconds=1.0))

        meter.update(1.0)
        meter.update(2.0)

        actual = meter.calculate_eta(time_diff)
        desired = datetime.now()
        self.assertAlmostEqual(actual, desired, delta=timedelta(seconds=1.0))

    def test_ETAMeter_str_smoke(self):
        for show_local_eta in (True, False):
            meter = optim.ETAMeter(1, show_local_eta=show_local_eta)
            self.assertIsInstance(str(meter), str)
            meter.update(0.0)
            self.assertIsInstance(str(meter), str)
            meter.update(0.0)
            self.assertIsInstance(str(meter), str)

    def test_ProgressMeter_update(self):
        meters = [optim.AverageMeter(str(idx)) for idx in range(3)]
        progress_meter = optim.ProgressMeter(1, *meters)

        vals = [float(val) for val in range(len(meters))]
        progress_meter.update(**{meter.name: val for meter, val in zip(meters, vals)})

        actual = progress_meter.count
        desired = 1
        self.assertEqual(actual, desired)

        for meter, val in zip(meters, vals):
            actual = meter.last_val
            desired = val
            self.assertFloatAlmostEqual(actual, desired)

    def test_ProgressMeter_str_smoke(self):
        for name in (None, "progress_meter"):
            meter = optim.ETAMeter(1, name=name)
            self.assertIsInstance(str(meter), str)
            meter.update(0.0)
            self.assertIsInstance(str(meter), str)


class TestOptim(PysticheTestCase):
    pass
