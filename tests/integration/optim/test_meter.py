from datetime import datetime, timedelta

import pytest

import torch

import pystiche
from pystiche import optim


class FloatMeter(optim.FloatMeter):
    def __init__(self, window_size=10):
        super().__init__("float_meter", window_size=window_size)

    def __str__(self):
        pass


def test_FloatMeter_count():
    torch.manual_seed(0)
    vals = torch.rand(10)

    meter = FloatMeter()
    meter.update(vals.tolist())

    actual = meter.count
    desired = len(vals)
    assert actual == desired


def test_FloatMeter_last_val():
    torch.manual_seed(0)
    vals = torch.rand(10)

    meter = FloatMeter()
    meter.update(vals.tolist())

    actual = meter.last_val
    desired = vals[-1].item()
    assert actual == pytest.approx(desired)


def test_FloatMeter_global_sum():
    torch.manual_seed(0)
    vals = torch.rand(10)

    meter = FloatMeter()
    meter.update(vals.tolist())

    actual = meter.global_sum
    desired = torch.sum(vals).item()
    assert actual == pytest.approx(desired)


def test_FloatMeter_global_min():
    torch.manual_seed(0)
    vals = torch.rand(10)

    meter = FloatMeter()
    meter.update(vals.tolist())

    actual = meter.global_min
    desired = torch.min(vals).item()
    assert actual == pytest.approx(desired)


def test_FloatMeter_global_max():
    torch.manual_seed(0)
    vals = torch.rand(10)

    meter = FloatMeter()
    meter.update(vals.tolist())

    actual = meter.global_max
    desired = torch.max(vals).item()
    assert actual == pytest.approx(desired)


def test_FloatMeter_global_avg():
    torch.manual_seed(0)
    vals = torch.rand(10)

    meter = FloatMeter()
    meter.update(vals.tolist())

    actual = meter.global_avg
    desired = torch.mean(vals).item()
    assert actual == pytest.approx(desired)


def test_FloatMeter_global_avg_empty():
    meter = FloatMeter()
    with pytest.raises(RuntimeError):
        meter.global_avg


def test_FloatMeter_local_avg():
    window_size = 5

    torch.manual_seed(0)
    vals = torch.rand(10)

    meter = FloatMeter(window_size=window_size)
    meter.update(vals.tolist())

    actual = meter.local_avg
    desired = torch.mean(vals[-window_size:]).item()
    assert actual == pytest.approx(desired)


def test_FloatMeter_local_avg_empty():
    meter = FloatMeter()
    with pytest.raises(RuntimeError):
        meter.local_avg


def test_AverageMeter_update_tensor():
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
        assert actual == pytest.approx(desired)


def test_AverageMeter_str_smoke():
    for show_local_avg in (True, False):
        meter = optim.AverageMeter("test_average_meter", show_local_avg=show_local_avg)
        assert isinstance(str(meter), str)
        meter.update(0.0)
        assert isinstance(str(meter), str)


def test_LossMeter_update_LossDict():
    actual_meter = optim.LossMeter("actual_meter")
    desired_meter = optim.LossMeter("desired_meter")

    losses = torch.arange(3, dtype=torch.float)
    loss_dict = pystiche.LossDict([(str(idx), loss) for idx, loss in enumerate(losses)])

    actual_meter.update(loss_dict)
    desired_meter.update(torch.sum(losses).item())

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
        assert actual == pytest.approx(desired)


def test_ETAMeter():
    first_time = 1.0
    second_time = 3.0
    meter = optim.ETAMeter(1)

    meter.update(first_time)

    actual = meter.count
    desired = 0
    assert actual == desired

    actual = meter.last_time
    desired = first_time
    assert actual == desired

    meter.update(second_time)

    actual = meter.count
    desired = 1
    assert actual == desired

    actual = meter.last_time
    desired = second_time
    assert actual == desired

    actual = meter.last_val
    desired = second_time - first_time
    assert actual == desired


def assert_datetime_almost_equal(actual, desired, delta):
    if isinstance(delta, float):
        delta = timedelta(seconds=delta)
    assert abs(actual - desired) <= delta


def test_ETAMeter_calculate_eta():
    meter = optim.ETAMeter(1)
    time_diff = 10.0
    delta = time_diff / 2

    actual = meter.calculate_eta(time_diff)
    desired = datetime.now() + timedelta(seconds=time_diff)
    assert_datetime_almost_equal(actual, desired, delta)

    meter.update(1.0)
    meter.update(2.0)

    actual = meter.calculate_eta(time_diff)
    desired = datetime.now()
    assert_datetime_almost_equal(actual, desired, delta)


def test_ETAMeter_str_smoke():
    for show_local_eta in (True, False):
        meter = optim.ETAMeter(1, show_local_eta=show_local_eta)
        assert isinstance(str(meter), str)
        meter.update(0.0)
        assert isinstance(str(meter), str)
        meter.update(0.0)
        assert isinstance(str(meter), str)


def test_ProgressMeter_update():
    meters = [optim.AverageMeter(str(idx)) for idx in range(3)]
    progress_meter = optim.ProgressMeter(1, *meters)

    vals = [float(val) for val in range(len(meters))]
    progress_meter.update(**{meter.name: val for meter, val in zip(meters, vals)})

    actual = progress_meter.count
    desired = 1
    assert actual == desired

    for meter, val in zip(meters, vals):
        actual = meter.last_val
        desired = val
        assert actual == pytest.approx(desired)


def test_ProgressMeter_str_smoke():
    for name in (None, "progress_meter"):
        meter = optim.ETAMeter(1, name=name)
        assert isinstance(str(meter), str)
        meter.update(0.0)
        assert isinstance(str(meter), str)
