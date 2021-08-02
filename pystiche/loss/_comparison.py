import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import torch
from torch import nn
from torchvision.transforms.functional import affine

import pystiche
from pystiche import enc
from pystiche._compat import InterpolationMode
from pystiche.misc import to_2d_arg

from . import functional as F
from ._loss import ComparisonLoss
from .utils import apply_guide

__all__ = [
    "FeatureReconstructionLoss",
    "GramLoss",
    "MRFLoss",
]


class FeatureReconstructionLoss(ComparisonLoss):
    r"""The feature reconstruction loss is the de facto standard content loss. It
    measures the mean squared error (MSE) between the encodings of an ``input_image``
    :math:`\hat{I}` and a ``target_image`` :math:`I` :

    .. math::

        \mean \parentheses{\Phi\of{\hat{I}} - \Phi\of{I}}^2

    Here :math:`\Phi\of{\cdot}` denotes the ``encoder``.

    .. note::

        Opposed to the paper, the implementation calculates the grand average
        :math:`\mean` opposed to the grand sum :math:`\sum` to account for different
        sized images.

    Args:
        encoder: Encoder :math:`\Phi`.
        score_weight: Score weight of the operator. Defaults to ``1.0``.

    Examples:

        >>> mle = pystiche.enc.vgg19_multi_layer_encoder()
        >>> encoder = mle.extract_encoder("relu4_2")
        >>> loss = pystiche.loss.FeatureReconstructionLoss(encoder)
        >>> input = torch.rand(2, 3, 256, 256)
        >>> target = torch.rand(2, 3, 256, 256)
        >>> loss.set_target_image(target)
        >>> score = loss(input)

    .. seealso::

        The feature reconstruction loss was introduced by Mahendran and Vedaldi in
        :cite:`MV2015` , but its name was coined by Johnson, Alahi, and Fei-Fei in
        :cite:`JAFF2016` .
    """

    def __init__(
        self,
        encoder: enc.Encoder,
        *,
        input_guide: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
        target_guide: Optional[torch.Tensor] = None,
        score_weight: float = 1e0,
    ):
        super().__init__(
            encoder=encoder,
            input_guide=input_guide,
            target_image=target_image,
            target_guide=target_guide,
            score_weight=score_weight,
        )

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return enc

    def input_enc_to_repr(
        self, enc: torch.Tensor, ctx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.enc_to_repr(enc)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.enc_to_repr(enc), None

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return F.mse_loss(input_repr, target_repr)


class GramLoss(ComparisonLoss):
    r"""The gram loss is a style loss based on the correlation of feature map channels.
    It measures the mean squared error (MSE) between the channel-wise Gram matrices of
    the encodings of an ``input_image`` :math:`\hat{I}` and a ``target_image``
    :math:`I`:

    .. math::

        \mean \parentheses{\fun{gram}{\Phi\of{\hat{I}}} - \fun{gram}{\Phi\of{I}}}^2

    Here :math:`\Phi\of{\cdot}` denotes the ``encoder`` and :math:`\fun{gram}{\cdot}`
    denotes :func:`pystiche.gram_matrix`.

    .. note::

        Opposed to the paper, the implementation calculates the grand average
        :math:`\mean` opposed to the grand sum :math:`\sum` to account for different
        sized images.

    Args:
        encoder: Encoder :math:`\Phi\left( \cdot \right)`.
        normalize: If ``True``, normalizes the Gram matrices to account for different
            sized images. See :func:`pystiche.gram_matrix` for details. Defaults to
            ``True``.
        score_weight: Score weight of the operator. Defaults to ``1.0``.

    Examples:

        >>> mle = pystiche.enc.vgg19_multi_layer_encoder()
        >>> encoder = mle.extract_encoder("relu4_2")
        >>> loss = pystiche.loss.GramLoss(encoder)
        >>> input = torch.rand(2, 3, 256, 256)
        >>> target = torch.rand(2, 3, 256, 256)
        >>> loss.set_target_image(target)
        >>> score = loss(input)

    .. seealso::

        The feature reconstruction loss was introduced by Gatys, Ecker, and Bethge
        in :cite:`GEB2016` .
    """

    def __init__(
        self,
        encoder: enc.Encoder,
        *,
        normalize: bool = True,
        input_guide: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
        target_guide: Optional[torch.Tensor] = None,
        score_weight: float = 1.0,
    ) -> None:
        super().__init__(
            encoder=encoder,
            input_guide=input_guide,
            target_image=target_image,
            target_guide=target_guide,
            score_weight=score_weight,
        )
        self.normalize = normalize

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return pystiche.gram_matrix(enc, normalize=self.normalize)

    def input_enc_to_repr(
        self, enc: torch.Tensor, ctx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.enc_to_repr(enc)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.enc_to_repr(enc), None

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return F.mse_loss(input_repr, target_repr)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        if not self.normalize:
            dct["normalize"] = self.normalize
        return dct


class ScaleAndRotate(pystiche.Module):
    def __init__(self, scale_factor: float, rotation_angle: float) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.rotation_angle = rotation_angle

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            affine(
                image,
                angle=self.rotation_angle,
                translate=[0, 0],
                scale=self.scale_factor,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
            ),
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["scale_factor"] = self.scale_factor
        dct["rotation_angle"] = self.rotation_angle
        return dct


class MRFLoss(ComparisonLoss):
    r"""The MRF loss is a style loss based on
    `Markov Random Fields (MRFs) <https://en.wikipedia.org/wiki/Markov_random_field>`_.
    It measures the mean squared error (MSE) between *neural patches* extracted from
    the encodings of an ``input_image`` :math:`\hat{I}` and a ``target_image``
    :math:`I`:

    .. math::

        \mean \parentheses{p_n\of{\Phi\of{\hat{I}}} - p_{MCS\of{n}}\of{\Phi\of{\hat{I}}}}^2

    Since the number of patches might be different for both images and the order of the
    patches does not correlate with the order of the enclosed style element, for each
    input neural patch :math:`n` a fitting target neural patch is to selected based on
    the maximum cosine similarity :math:`MCS\of{n}` with
    :func:`pystiche.cosine_similarity`.

    .. note::

        Opposed to the paper, the implementation calculates the grand average
        :math:`\mean` opposed to the grand sum :math:`\sum` to account for different
        sized images.

    Args:
        encoder: Encoder :math:`\Phi` .
        patch_size: Spatial size of the neural patches.
        stride: Distance between two neural patches.
        target_transforms: Optional transformations to apply to the target image before
            the neural patches are extracted. Defaults to ``None``.
        score_weight: Score weight of the operator. Defaults to ``1.0``.

    Examples:

        >>> mle = pystiche.enc.vgg19_multi_layer_encoder()
        >>> encoder = mle.extract_encoder("relu4_2")
        >>> patch_size = 3
        >>> loss = pystiche.loss.MRFLoss(encoder, patch_size)
        >>> input = torch.rand(2, 3, 256, 256)
        >>> target = torch.rand(2, 3, 256, 256)
        >>> loss.set_target_image(target)
        >>> score = loss(input)

    .. seealso::

        The MRF loss was introduced by Li and Wand in :cite:`LW2016`.
    """

    def __init__(
        self,
        encoder: enc.Encoder,
        patch_size: Union[int, Sequence[int]],
        *,
        stride: Union[int, Sequence[int]] = 1,
        target_transforms: Optional[Iterable[nn.Module]] = None,
        input_guide: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
        target_guide: Optional[torch.Tensor] = None,
        score_weight: float = 1.0,
    ):
        super().__init__(
            encoder=encoder,
            input_guide=input_guide,
            target_image=target_image,
            target_guide=target_guide,
            score_weight=score_weight,
        )
        self.patch_size = to_2d_arg(patch_size)
        self.stride = to_2d_arg(stride)
        self.target_transforms = target_transforms

    @staticmethod
    def scale_and_rotate_transforms(
        num_scale_steps: int = 1,
        scale_step_width: float = 5e-2,
        num_rotate_steps: int = 1,
        rotate_step_width: float = 10.0,
    ) -> List[ScaleAndRotate]:
        """Generate a list of scaling and rotations transformations.

        .. seealso::

            The output of this method can be used as parameter ``target_transforms`` of
            :class:`~pystiche.loss.MRFLoss` to enrich the space of target neural
            patches:

            .. code-block:: python

                target_transforms = MRFOperator.scale_and_rotate_transforms()
                loss = pystiche.loss.MRFLoss(..., target_transforms=target_transforms)

        Args:
            num_scale_steps: Number of scale steps. Each scale is performed in both
                directions, i.e. enlarging and shrinking the motif. Defaults to ``1``.
            scale_step_width: Width of each scale step. Defaults to ``5e-2``.
            num_rotate_steps: Number of rotate steps. Each rotate is performed in both
                directions, i.e. clockwise and counterclockwise. Defaults to ``1``.
            rotate_step_width: Width of each rotation step in degrees.
                Defaults to ``10.0``.

        Returns:
           ``(num_scale_steps * 2 + 1) * (num_rotate_steps * 2 + 1)`` transformations
           in total comprising every combination given by the input parameters.
        """

        angles = [
            base * rotate_step_width
            for base in range(-num_rotate_steps, num_rotate_steps + 1)
        ]
        scale_factors = [
            1.0 + (base * scale_step_width)
            for base in range(-num_scale_steps, num_scale_steps + 1)
        ]
        return [
            ScaleAndRotate(scale_factor=scale_factor, rotation_angle=angle,)
            for angle, scale_factor in itertools.product(angles, scale_factors)
        ]

    def _guide_repr(self, repr: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Due to the guiding large areas of the images might be zero and thus many
        # patches might carry no information. These patches can be removed from the
        # target and input representation reducing the computing cost and memory during
        # the loss calculation.

        # Patches without information have constant values in the spatial dimensions.
        repr_flat = torch.flatten(repr, 3)
        constant = repr_flat[..., 0].unsqueeze(3)

        # By checking where the spatial values do not differ from this constant in any
        # channel, the patches with no information can be filtered out.
        abs_diff = torch.abs(repr_flat - constant)
        mask = torch.any(torch.flatten(abs_diff > eps, 2), dim=2)

        return repr[mask].unsqueeze(0)

    def enc_to_repr(self, enc: torch.Tensor, *, is_guided: bool) -> torch.Tensor:
        repr = pystiche.extract_patches2d(enc, self.patch_size, self.stride)
        if not is_guided:
            return repr

        return self._guide_repr(repr)

    def input_enc_to_repr(
        self, enc: torch.Tensor, ctx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.enc_to_repr(enc, is_guided=self._input_guide is not None)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.enc_to_repr(enc, is_guided=self._target_guide is not None), None

    def _target_image_to_repr(
        self, image: torch.Tensor, guide: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if guide is not None:
            # Due to the possible scaling and rotation, we only apply the guide to
            # the target image and not its encodings
            image = apply_guide(image, guide)

        if self.target_transforms is None:
            return self.target_enc_to_repr(self.encoder(image))  # type: ignore[misc]

        device = image.device
        reprs = []
        for transform in self.target_transforms:
            transform = transform.to(device)
            enc = self.encoder(transform(image))  # type: ignore[misc]
            repr, _ = self.target_enc_to_repr(enc)
            reprs.append(repr)

        repr = torch.cat(reprs)
        ctx = None
        return repr, ctx

    def set_target_image(
        self, image: torch.Tensor, guide: Optional[torch.Tensor] = None
    ) -> None:

        self.register_buffer("_target_image", image, persistent=True)
        self.register_buffer("_target_guide", guide, persistent=True)

        with torch.no_grad():
            repr, ctx = self._target_image_to_repr(image, guide)

        self.register_buffer("_target_repr", repr, persistent=True)
        self.register_buffer("_ctx", repr, persistent=True)

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return F.mrf_loss(input_repr, target_repr, batched_input=True)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["patch_size"] = self.patch_size
        dct["stride"] = self.stride
        return dct
