import pytest

from pystiche.image import transforms


def test_Transform_add():
    class TestTransform(transforms.Transform):
        def forward(self):
            pass

    def add_transforms(transforms):
        added_transform = transforms[0]
        for transform in transforms[1:]:
            added_transform += transform
        return added_transform

    test_transforms = (TestTransform(), TestTransform(), TestTransform())
    added_transform = add_transforms(test_transforms)

    assert isinstance(added_transform, transforms.ComposedTransform)
    for idx, test_transform in enumerate(test_transforms):
        actual = getattr(added_transform, str(idx))
        desired = test_transform
        assert actual is desired


def test_ComposedTransform_call():
    class Plus(transforms.Transform):
        def __init__(self, plus):
            super().__init__()
            self.plus = plus

        def forward(self, input):
            return input + self.plus

    num_transforms = 3
    composed_transform = transforms.ComposedTransform(
        *[Plus(plus) for plus in range(1, num_transforms + 1)]
    )

    actual = composed_transform(0)
    desired = num_transforms * (num_transforms + 1) // 2
    assert actual == desired


def test_ComposedTransform_add():
    class TestTransform(transforms.Transform):
        def forward(self):
            pass

    test_transforms = (TestTransform(), TestTransform(), TestTransform())
    composed_transform = transforms.ComposedTransform(*test_transforms[:-1])
    single_transform = test_transforms[-1]
    added_transform = composed_transform + single_transform

    assert isinstance(added_transform, transforms.ComposedTransform)
    for idx, test_transform in enumerate(test_transforms):
        actual = getattr(added_transform, str(idx))
        desired = test_transform
        assert actual is desired


def test_compose_transforms_other():
    with pytest.raises(TypeError):
        transforms.core.compose_transforms(None)
