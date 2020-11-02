import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import torch

import pystiche
from pystiche.enc import Encoder
from pystiche.image.transforms import Transform, TransformMotifAffinely
from pystiche.misc import suppress_warnings, to_2d_arg

from . import functional as F
from .op import EncodingComparisonOperator

__all__ = [
    "FeatureReconstructionOperator",
    "GramOperator",
    "MRFOperator",
]


class FeatureReconstructionOperator(EncodingComparisonOperator):
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

        >>> multi_layer_encoder = enc.vgg19_multi_layer_encoder()
        >>> encoder = multi_layer_encoder.extract_encoder("relu4_2")
        >>> op = ops.FeatureReconstructionOperator(encoder)
        >>> input = torch.rand(2, 3, 256, 256)
        >>> target = torch.rand(2, 3, 256, 256)
        >>> op.set_target_image(target)
        >>> score = op(input)

    .. seealso::

        The feature reconstruction loss was introduced by Mahendran and Vedaldi in
        :cite:`MV2015` , but its name was coined by Johnson, Alahi, and Fei-Fei in
        :cite:`JAFF2016` .
    """

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


class GramOperator(EncodingComparisonOperator):
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

        >>> multi_layer_encoder = enc.vgg19_multi_layer_encoder()
        >>> encoder = multi_layer_encoder.extract_encoder("relu4_2")
        >>> op = ops.GramOperator(encoder)
        >>> input = torch.rand(2, 3, 256, 256)
        >>> target = torch.rand(2, 3, 256, 256)
        >>> op.set_target_image(target)
        >>> score = op(input)

    .. seealso::

        The feature reconstruction loss was introduced by Gatys, Ecker, and Bethge
        in :cite:`GEB2016` .
    """

    def __init__(
        self, encoder: Encoder, normalize: bool = True, score_weight: float = 1.0
    ) -> None:
        super().__init__(encoder, score_weight=score_weight)
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


class MRFOperator(EncodingComparisonOperator):
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

        >>> multi_layer_encoder = enc.vgg19_multi_layer_encoder()
        >>> encoder = multi_layer_encoder.extract_encoder("relu4_2")
        >>> patch_size = 3
        >>> op = ops.MRFOperator(encoder, patch_size)
        >>> input = torch.rand(2, 3, 256, 256)
        >>> target = torch.rand(2, 3, 256, 256)
        >>> op.set_target_image(target)
        >>> score = op(input)

    .. seealso::

        The MRF loss was introduced by Li and Wand in :cite:`LW2016`.
    """

    def __init__(
        self,
        encoder: Encoder,
        patch_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        target_transforms: Optional[Iterable[Transform]] = None,
        score_weight: float = 1.0,
    ):
        super().__init__(encoder, score_weight=score_weight)
        self.patch_size = to_2d_arg(patch_size)
        self.stride = to_2d_arg(stride)
        self.target_transforms = target_transforms

    @staticmethod
    def scale_and_rotate_transforms(
        num_scale_steps: int = 1,
        scale_step_width: float = 5e-2,
        num_rotate_steps: int = 1,
        rotate_step_width: float = 10.0,
    ) -> List[TransformMotifAffinely]:
        """Generate a list of scaling and rotations transformations.

        .. seealso::

            The output of this method can be used as parameter ``target_transforms`` of
            :class:`~pystiche.ops.MRFOperator` to enrich the space of target neural
            patches:

            .. code-block:: python

                from pystiche.ops import MRFOperator


                target_transforms = MRFOperator.scale_and_rotate_transforms()
                op = MRFOperator(..., target_transforms=target_transforms)

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
        scaling_factors = np.arange(
            -num_scale_steps, num_scale_steps + 1, dtype=np.float
        )
        scaling_factors = 1.0 + (scaling_factors * scale_step_width)

        rotation_angles = np.arange(
            -num_rotate_steps, num_rotate_steps + 1, dtype=np.float
        )
        rotation_angles *= rotate_step_width

        transforms = []
        for scaling_factor, rotation_angle in itertools.product(
            scaling_factors, rotation_angles
        ):
            with suppress_warnings():
                transform = TransformMotifAffinely(
                    scaling_factor=scaling_factor,
                    rotation_angle=rotation_angle,
                    canvas="same",  # FIXME: this should be valid after it is implemented
                )
            transforms.append(transform)

        return transforms

    @staticmethod
    def _match_batch_sizes(target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        # FIXME
        return target

    def set_target_guide(self, guide: torch.Tensor, recalc_repr: bool = True) -> None:
        # Since the target representation of the MRFOperator possibly comprises
        # scaled or rotated patches, it is not useful to store the target encoding
        # guides
        self.register_buffer("target_guide", guide)
        if recalc_repr and self.has_target_image:
            self.set_target_image(self.target_image)

    def _guide_repr(self, repr: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Due to the guiding large areas of the images might be zero and thus many
        # patches might carry no information. These patches can be removed from the
        # target and input representation reducing the computing cost and memory during
        # the loss calculation.

        # Patches without information have constant values in the spatial dimensions.
        repr_flat = torch.flatten(repr, 2)
        constant = repr_flat[:, :, 0].unsqueeze(2)

        # By checking where the spatial values do not differ from this constant in any
        # channel, the patches with no information can be filtered out.
        abs_diff = torch.abs(repr_flat - constant)
        mask = torch.any(torch.flatten(abs_diff > eps, 1), dim=1)

        return repr[mask]

    def enc_to_repr(self, enc: torch.Tensor, is_guided: bool) -> torch.Tensor:
        with suppress_warnings(FutureWarning):
            repr = pystiche.extract_patches2d(enc, self.patch_size, self.stride)
        if not is_guided:
            return repr

        return self._guide_repr(repr)

    def input_enc_to_repr(
        self, enc: torch.Tensor, ctx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.enc_to_repr(enc, self.has_input_guide)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.enc_to_repr(enc, self.has_target_guide), None

    def target_image_to_repr(self, image: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Due to the possible scaling and rotation, we only apply the guide to the
        # target image and not the encodings
        if self.has_target_guide:
            image = self.apply_guide(image, self.target_guide)

        if self.target_transforms is None:
            return self.target_enc_to_repr(self.encoder(image))

        device = image.device
        reprs = []
        for transform in self.target_transforms:
            transform = transform.to(device)
            enc = self.encoder(transform(image))
            repr, _ = self.target_enc_to_repr(enc)
            reprs.append(repr)

        repr = torch.cat(reprs)
        ctx = None
        return repr, ctx

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return F.mrf_loss(input_repr, target_repr, batched_input=False)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["patch_size"] = self.patch_size
        dct["stride"] = self.stride
        return dct
