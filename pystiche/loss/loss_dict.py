from collections import OrderedDict


class LossDict(OrderedDict):
    def backward(self) -> None:
        sum(self.values()).backward()
