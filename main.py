from torch import nn


class Foo:
    def __init__(self, bar):
        self.bar = bar


class Baz(Foo, nn.Module):
    pass


baz = Baz(nn.Conv2d(3, 3, 1))
