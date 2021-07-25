from typing import Dict, Iterable, Optional, Tuple

import torch

from pystiche.loss import ComparisonLoss, Loss

__all__ = ["ImageStorage"]


class ImageStorage:
    def __init__(self, losses: Iterable[Loss]) -> None:
        self.target_images_and_guides: Dict[
            ComparisonLoss, Tuple[torch.Tensor, Optional[torch.Tensor]]
        ] = {}
        self.input_guides: Dict[Loss, torch.Tensor] = {}
        for loss in losses:
            if isinstance(loss, ComparisonLoss):
                if loss.target_image is not None:
                    self.target_images_and_guides[loss] = (
                        loss.target_image,
                        loss.target_guide,
                    )

            if loss.input_guide is not None:
                self.input_guides[loss] = loss.input_guide

    def restore(self) -> None:
        for loss, (target_image, target_guide) in self.target_images_and_guides.items():
            loss.set_target_image(target_image, guide=target_guide)

        for loss, input_guide in self.input_guides.items():
            loss.set_input_guide(input_guide)
