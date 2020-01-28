from abc import abstractmethod
from typing import Any, Dict, NoReturn
import torch
from torch import nn


class Module(nn.Module):
    _STR_INDENT = 2

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Any:
        pass

    def _build_str(self, name=None, description=None, named_children=None):
        if name is None:
            name = self.__class__.__name__

        if description is None:
            description = self.description()

        if named_children is None:
            named_children = tuple(self.named_children())

        prefix = f"{name}("
        postfix = ")"

        description_lines = description.splitlines()
        multi_line_descr = len(description_lines) > 1

        if not named_children and not multi_line_descr:
            return prefix + description + postfix

        def indent(line):
            return " " * self._STR_INDENT + line

        body = []
        for line in description_lines:
            body.append(indent(line))

        for name, module in named_children:
            lines = str(module).splitlines()
            body.append(indent(f"({name}): {lines[0]}"))
            for line in lines[1:]:
                body.append(indent(line))

        return "\n".join([prefix] + body + [postfix])

    def __str__(self) -> str:
        return self._build_str()

    def description(self) -> str:
        return ""

    def extra_repr(self) -> str:
        return self.description()


class TensorStorage(nn.Module):
    def __init__(self, **attrs: Dict[str, Any]) -> None:
        super().__init__()
        for name, attr in attrs.items():
            if isinstance(attr, torch.Tensor):
                self.register_buffer(name, attr)
            else:
                setattr(self, name, attr)

    def forward(self) -> NoReturn:
        msg = (
            f"{self.__class__.__name__} objects are only used "
            "for storage and cannot be called."
        )
        raise RuntimeError(msg)
