from typing import Dict, Iterable

import torch

from pystiche.ops import ComparisonOperator, Operator

__all__ = ["ImageStorage"]


class ImageStorage:
    def __init__(self, ops: Iterable[Operator]) -> None:
        self.target_guides: Dict[ComparisonOperator, torch.Tensor] = {}
        self.target_images: Dict[ComparisonOperator, torch.Tensor] = {}
        self.input_guides: Dict[Operator, torch.Tensor] = {}
        for op in ops:
            if isinstance(op, ComparisonOperator):
                if op.has_target_guide:
                    self.target_guides[op] = op.target_guide

                if op.has_target_image:
                    self.target_images[op] = op.target_image

            if op.has_input_guide:
                self.input_guides[op] = op.input_guide

    def restore(self) -> None:
        for op, target_guide in self.target_guides.items():
            op.set_target_guide(target_guide, recalc_repr=False)

        for op, target_image in self.target_images.items():
            op.set_target_image(target_image)

        for op, input_guide in self.input_guides.items():
            op.set_input_guide(input_guide)
