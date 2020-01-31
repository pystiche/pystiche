from typing import Optional

__all__ = ["get_align_corners"]


# this method should only be used as long as interpolate(), grid_sample(), and
# affine_grid() from torch.nn.functional raise a UserWarning for changed default
# behaviour
def get_align_corners(interpolation_mode: str) -> Optional[bool]:
    if interpolation_mode in ("nearest", "area"):
        return None
    else:
        return False
