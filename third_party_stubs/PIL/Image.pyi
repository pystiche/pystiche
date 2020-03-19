from typing import Any, Union, Optional
from io import FileIO
import pathlib

class Image:
    def save(
        self,
        fp: Union[str, pathlib.Path, FileIO],
        format: Optional[str] = ...,
        **params: Any
    ): ...
    def show(self, title: Optional[str] = ..., command: Optional[str] = ...): ...

def open(fp: Union[str, pathlib.Path, FileIO]) -> Image: ...
