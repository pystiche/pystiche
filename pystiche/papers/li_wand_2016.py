from collections import OrderedDict
import torch
from torch import optim
import pystiche
from pystiche.misc import to_engstr
from pystiche.encoding import vgg19_encoder
from pystiche.nst import (
    DirectEncodingComparisonOperator,
    MRFEncodingComparisonOperator,
    TotalVariationPixelRegularizationOperator,
    MultiOperatorEncoder,
    CaffePreprocessingImageOptimizer,
    ImageOptimizerOctavePyramid,
)
import pystiche.nst.functional as F

__all__ = [
    "LiWand2016ContentLoss",
    "LiWand2016StyleLoss",
    "LiWand2016Regularizer",
    "LiWand2016NST",
    "LiWand2016NSTPyramid",
]


def get_encoder():
    return vgg19_encoder(weights="caffe", preprocessing=False)


def optimizer_getter(input_image):
    return optim.LBFGS([input_image], lr=1.0, max_iter=1)


class LiWand2016ContentLoss(DirectEncodingComparisonOperator):
    r"""Content loss from
    `"Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis" <https://ieeexplore.ieee.org/document/7780641>`

    Args:
        encoder: Encoder used to generate the feature maps. If None, a VGG19 encoder
            with weights from _Caffe_ and without internal preprocessing is used.
        impl_params: If True, hyper parameters from the authors implementation
            <https://github.com/chuanli11/CNNMRF> rather than the parameters given in
            the paper are used.
    """

    def __init__(self, encoder=None, impl_params=True):
        if encoder is None:
            encoder = get_encoder()
        self.impl_params = impl_params

        name = "Content loss (direct)"
        encoding_layers = ("relu_4_2",)
        layer_weights = "sum"
        score_weight = 2e1 if impl_params else 1e0
        super().__init__(
            encoder,
            encoding_layers,
            name=name,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )

        self.loss_reduction = "mean" if impl_params else "sum"

    def _calculate_score(self, input_repr, target_repr, ctx):
        return F.mse_loss(input_repr, target_repr, reduction=self.loss_reduction)

    def extra_descriptions(self):
        dct = OrderedDict()
        dct["Implementation parameters"] = self.impl_params
        if self.loss_reduction != "mean":
            dct["Loss reduction"] = self.loss_reduction
        return dct


class NormalizeUnfoldGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, size, step):
        ctx.needs_normalizing = step < size
        if ctx.needs_normalizing:
            normalizer = torch.zeros_like(input)
            item = [slice(None) for _ in range(input.dim())]
            for idx in range(0, normalizer.size()[dim] - size, step):
                item[dim] = slice(idx, idx + size)
                normalizer[item].add_(1.0)

            # clamping to avoid zero division
            ctx.save_for_backward(torch.clamp(normalizer, min=1.0))
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_normalizing:
            normalizer, = ctx.saved_tensors
            grad_input = grad_output / normalizer
        else:
            grad_input = grad_output.clone()
        return grad_input, None, None, None


normalize_unfold_grad = NormalizeUnfoldGrad.apply


def extract_normalized_patches2d(input, patch_size, stride):
    for dim, size, step in zip(range(2, input.dim()), patch_size, stride):
        input = normalize_unfold_grad(input, dim, size, step)
    return pystiche.extract_patches2d(input, patch_size, stride)


class LiWand2016StyleLoss(MRFEncodingComparisonOperator):
    r"""Style loss based on matching neural patches from
    `"Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis" <https://ieeexplore.ieee.org/document/7780641>`

    Args:
        encoder: Encoder used to generate the feature maps. If None, a VGG19 encoder
            with weights from _Caffe_ and without internal preprocessing is used.
        impl_params: If True, hyper parameters from the authors implementation
            <https://github.com/chuanli11/CNNMRF> rather than the parameters given in
            the paper are used.
    """

    def __init__(self, encoder=None, impl_params=True):
        if encoder is None:
            encoder = get_encoder()
        self.impl_params = impl_params

        name = "Style loss (MRF)"
        encoding_layers = ("relu_3_1", "relu_4_1")
        patch_size = 3
        stride = 2 if impl_params else 1
        if impl_params:
            num_scale_steps = 1
            num_rotation_steps = 1
        else:
            num_scale_steps = 3
            num_rotation_steps = 2
        scale_step_width = 5e-2
        rotation_step_width = 7.5
        layer_weights = "sum"
        score_weight = 1e-4 if impl_params else 1e0
        super().__init__(
            encoder,
            encoding_layers,
            patch_size,
            name=name,
            stride=stride,
            num_scale_steps=num_scale_steps,
            num_rotation_steps=num_rotation_steps,
            scale_step_width=scale_step_width,
            rotation_step_width=rotation_step_width,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )

        self.normalize_patches_grad = impl_params
        self._calculate_mrf_repr = (
            extract_normalized_patches2d
            if self.normalize_patches_grad
            else pystiche.extract_patches2d
        )

        self.loss_reduction = "sum"
        self.score_correction_factor = 1.0 / 2.0 if impl_params else 1.0

    def _calculate_score(self, input_repr, target_repr, ctx):
        score = F.patch_matching_loss(
            input_repr, target_repr, reduction=self.loss_reduction
        )
        return score * self.score_correction_factor

    def extra_descriptions(self):
        dct = OrderedDict()
        dct["Implementation parameters"] = self.impl_params
        if self.normalize_patches_grad:
            dct["Normalize patches gradient"] = self.normalize_patches_grad
        dct["Loss reduction"] = self.loss_reduction
        if self.score_correction_factor != 1.0:
            dct["Score correction factor"] = to_engstr(self.score_correction_factor)
        return dct


class LiWand2016Regularizer(TotalVariationPixelRegularizationOperator):
    r"""Image regularizer based on total variation from
    `"Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis" <https://ieeexplore.ieee.org/document/7780641>`

    Args:
        impl_params: If True, hyper parameters from the authors implementation
            <https://github.com/chuanli11/CNNMRF> rather than the parameters given in
            the paper are used.
    """

    def __init__(self, impl_params=True):
        self.impl_params = impl_params

        name = "Regularizer (total variation)"
        exponent = 2.0
        score_weight = 1e-3
        super().__init__(exponent=exponent, name=name, score_weight=score_weight)

        self.loss_reduction = "sum"
        self.score_correction_factor = 1.0 / 2.0 if impl_params else 1.0

    def _calculate_score(self, input_repr):
        score = F.total_variation_loss(
            input_repr, exponent=self.exponent, reduction=self.loss_reduction
        )
        return score * self.score_correction_factor

    def extra_descriptions(self):
        dct = OrderedDict()
        dct["Implementation parameters"] = self.impl_params
        dct["Loss reduction"] = self.loss_reduction
        if self.score_correction_factor != 1.0:
            dct["Score correction factor"] = to_engstr(self.score_correction_factor)
        return dct


class LiWand2016NST(CaffePreprocessingImageOptimizer):
    r"""Image regularizer based on total variation from
    `"Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis" <https://ieeexplore.ieee.org/document/7780641>`

    A VGG19 encoder with weights from _Caffe_ and without internal preprocessing is
    used. The image is optimized with LBFGS.

    Args:
        impl_params: If True, hyper parameters from the authors implementation
            <https://github.com/chuanli11/CNNMRF> rather than the parameters given in
            the paper are used.

    Attributes:
        content_loss: Content loss operator
        style_loss: Style loss operator
        regularizer: Regularization operator
    """

    def __init__(self, impl_params=True):
        encoder = MultiOperatorEncoder(get_encoder())
        content_loss = LiWand2016ContentLoss(encoder)
        style_loss = LiWand2016StyleLoss(encoder, impl_params)
        regularizer = LiWand2016Regularizer(impl_params)

        super().__init__(
            content_loss, style_loss, regularizer, optimizer_getter=optimizer_getter
        )
        self.content_loss = content_loss
        self.style_loss = style_loss
        self.regularizer = regularizer


class LiWand2016NSTPyramid(ImageOptimizerOctavePyramid):
    r"""Image regularizer based on total variation from
    `"Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis" <https://ieeexplore.ieee.org/document/7780641>`

    A VGG19 encoder with weights from _Caffe_ and without internal preprocessing is
    used. The image is optimized with LBFGS.

    The pyramid comprises two levels with 500 and 200 steps respectively. The images
    are resized so that their short edge is 500 and 800 pixels wide.

    Args:
        impl_params: If True, hyper parameters from the authors implementation
            <https://github.com/chuanli11/CNNMRF> rather than the parameters given in
            the paper are used.

    Attributes:
        nst: NST image optimizer
    """

    def __init__(self, impl_params=True):
        nst = LiWand2016NST(impl_params)
        super().__init__(nst)
        self.nst = nst
        self.impl_params = impl_params
        self.build_levels()

    def build_levels(self):
        """
        Build the levels of the pyramid. The image size between two levels is increased
        by a factor of two. The pyramid starts with an image, which longest edge is
        atleast 64 pixels wide. On each level 100 optimization steps are performed.
        """
        max_edge_size = 384
        level_steps = 100 if self.impl_params else 200
        num_levels = 3 if self.impl_params else None
        min_edge_size = 64
        edges = "long"
        super().build_levels(
            max_edge_size,
            level_steps,
            num_levels,
            min_edge_size=min_edge_size,
            edges=edges,
        )
