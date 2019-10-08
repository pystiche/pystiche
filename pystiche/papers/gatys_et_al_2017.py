from collections import OrderedDict
from torch import optim
import pystiche
from pystiche.misc import to_engstr
from pystiche.image import extract_aspect_ratio
from pystiche.encoding import vgg19_encoder
from pystiche.nst import (
    DirectEncodingComparisonOperator,
    GramEncodingComparisonOperator,
    GuidedEncodingComparisonOperator,
    MultiOperatorEncoder,
    CaffePreprocessingImageOptimizer,
    ImageOptimizerPyramid,
)

__all__ = [
    "GatysEtAl2017ContentLoss",
    "GatysEtAl2017StyleLoss",
    "GatysEtAl2017NST",
    "GatysEtAl2017NSTPyramid",
    "GatysEtAl2017SpatialControlStyleLoss",
    "GatysEtAl2017SpatialControlNST",
    "GatysEtAl2017SpatialControlNSTPyramid",
]


def get_encoder():
    return vgg19_encoder(weights="caffe", preprocessing=False)


def optimizer_getter(input_image):
    return optim.LBFGS([input_image], lr=1.0, max_iter=1)


class GatysEtAl2017ContentLoss(DirectEncodingComparisonOperator):
    r"""Content loss from
    `"Controlling Perceptual Factors in Neural Style Transfer" <http://openaccess.thecvf.com/content_cvpr_2017/papers/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.pdf>`

    Args:
        encoder: Encoder used to generate the feature maps. If None, a VGG19 encoder
            with weights from _Caffe_ and without internal preprocessing is used.
    """

    def __init__(self, encoder=None):
        if encoder is None:
            encoder = get_encoder()
        name = "Content loss (direct)"
        layers = ("relu_4_2",)
        score_weight = 1e0
        super().__init__(encoder, layers, name=name, score_weight=score_weight)


class GatysEtAl2017StyleLoss(GramEncodingComparisonOperator):
    r"""Style loss based on gram matrices from
    `"Controlling Perceptual Factors in Neural Style Transfer" <http://openaccess.thecvf.com/content_cvpr_2017/papers/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.pdf>`

    Args:
        encoder: Encoder used to generate the feature maps. If None, a VGG19 encoder
            with weights from _Caffe_ and without internal preprocessing is used.
        impl_params: If True, hyper parameters from the authors implementation
            <https://github.com/leongatys/PytorchNeuralStyleTransfer> rather than the
            parameters given in the paper are used.
    """

    def __init__(self, encoder=None, impl_params=True):
        if encoder is None:
            encoder = get_encoder()
        self.impl_params = impl_params

        name = "Style loss (Gram)"
        layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
        layer_weights = [
            1.0 / num_channels ** 2.0 for num_channels in (64, 128, 256, 512, 512)
        ]
        normalize = True
        score_weight = 1e3
        super().__init__(
            encoder,
            layers,
            normalize=normalize,
            name=name,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )

        self.score_correction_factor = 1.0 if impl_params else 1.0 / 4.0

    def _calculate_score(self, *args, **kwargs):
        return super()._calculate_score(*args, **kwargs) * self.score_correction_factor

    def extra_descriptions(self):
        dct = OrderedDict()
        dct["Implementation parameters"] = self.impl_params
        if self.score_correction_factor != 1.0:
            dct["Score correction factor"] = to_engstr(self.score_correction_factor)
        return dct


class GatysEtAl2017SpatialControlStyleLoss(
    GuidedEncodingComparisonOperator, GatysEtAl2017StyleLoss
):
    r"""Spatially guided style loss based on gram matrices from
    `"Controlling Perceptual Factors in Neural Style Transfer" <http://openaccess.thecvf.com/content_cvpr_2017/papers/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.pdf>`

    Args:
        encoder: Encoder used to generate the feature maps. If None, a VGG19 encoder
            with weights from _Caffe_ and without internal preprocessing is used.
        impl_params: If True, hyper parameters from the authors implementation
            <https://github.com/leongatys/PytorchNeuralStyleTransfer> rather than the
            parameters given in the paper are used.
    """


class GatysEtAl2017NST(CaffePreprocessingImageOptimizer):
    r"""NST from
    `"Controlling Perceptual Factors in Neural Style Transfer" <http://openaccess.thecvf.com/content_cvpr_2017/papers/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.pdf>`

    A VGG19 encoder with weights from _Caffe_ and without internal preprocessing is
    used. The image is optimized with LBFGS.

    Args:
        impl_params: If True, hyper parameters from the authors implementation
            <https://github.com/leongatys/PytorchNeuralStyleTransfer> rather than the
            parameters given in the paper are used.

    Attributes:
        content_loss: Content loss operator
        style_loss: Style loss operator
    """

    def __init__(self, impl_params=True):
        encoder = MultiOperatorEncoder(get_encoder())
        content_loss = GatysEtAl2017ContentLoss(encoder)
        style_loss = GatysEtAl2017StyleLoss(encoder, impl_params)

        super().__init__(content_loss, style_loss, optimizer_getter=optimizer_getter)
        self.content_loss = content_loss
        self.style_loss = style_loss


class GatysEtAl2017SpatialControlNST(CaffePreprocessingImageOptimizer):
    r"""Spatially guided NST from
    `"Controlling Perceptual Factors in Neural Style Transfer" <http://openaccess.thecvf.com/content_cvpr_2017/papers/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.pdf>`

    A VGG19 encoder with weights from _Caffe_ and without internal preprocessing is
    used. The image is optimized with LBFGS.

    Args:
        num_guides: Number of guides
        impl_params: If True, hyper parameters from the authors implementation
            <https://github.com/leongatys/PytorchNeuralStyleTransfer> rather than the
            parameters given in the paper are used.
        guide_names: Optional names attached to the individual style loss operator.

    Attributes:
        content_loss: Content loss operator
        style_losses: Tuple of spatially guided style loss operators
    """

    def __init__(self, num_guides, impl_params=True, guide_names=None):
        if guide_names is None:
            guide_names = [str(idx) for idx in range(num_guides)]
        else:
            assert len(guide_names) == num_guides

        encoder = MultiOperatorEncoder(get_encoder())
        content_loss = GatysEtAl2017ContentLoss(encoder)
        style_losses = []
        for guide_name in guide_names:
            style_loss = GatysEtAl2017SpatialControlStyleLoss(
                encoder, impl_params=impl_params
            )
            style_loss.name += " ({})".format(guide_name)
            style_losses.append(style_loss)

        super().__init__(content_loss, *style_losses, optimizer_getter=optimizer_getter)
        self.content_loss = content_loss
        self.style_losses = pystiche.tuple(style_losses)


class _GatysEtAl2017NSTPyramidBase(ImageOptimizerPyramid):
    def __init__(self, nst, impl_params):
        super().__init__(nst)
        self.nst = nst
        self.impl_params = impl_params
        self.build_levels()

    def build_levels(self):
        """
        Build the levels of the pyramid. The pyramid comprises two levels with 500 and
        200 steps respectively. The images are resized so that their short edge is 500
        and 800 pixels wide.
        """
        level_edge_sizes = 512 if self.impl_params else 500, 800
        level_steps = 500, 200
        edges = "short"
        super().build_levels(level_edge_sizes, level_steps, edges=edges)


class GatysEtAl2017NSTPyramid(_GatysEtAl2017NSTPyramidBase):
    r"""NST pyramid from
    `"Controlling Perceptual Factors in Neural Style Transfer" <http://openaccess.thecvf.com/content_cvpr_2017/papers/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.pdf>`

    A VGG19 encoder with weights from _Caffe_ and without internal preprocessing is
    used. The image is optimized with LBFGS.

    Args:
        impl_params: If True, hyper parameters from the authors implementation
            <https://github.com/leongatys/PytorchNeuralStyleTransfer> rather than the
            parameters given in the paper are used.

    Attributes:
        nst: NST image optimizer
    """

    def __init__(self, impl_params=True):
        nst = GatysEtAl2017NST(impl_params)
        super().__init__(nst, impl_params)


class GatysEtAl2017SpatialControlNSTPyramid(_GatysEtAl2017NSTPyramidBase):
    r"""Spatially guided NST pyramid from
    `"Controlling Perceptual Factors in Neural Style Transfer" <http://openaccess.thecvf.com/content_cvpr_2017/papers/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.pdf>`

    A VGG19 encoder with weights from _Caffe_ and without internal preprocessing is
    used. The image is optimized with LBFGS.

    Args:
        num_guides: Number of guides
        impl_params: If True, hyper parameters from the authors implementation
            <https://github.com/leongatys/PytorchNeuralStyleTransfer> rather than the
            parameters given in the paper are used.
        guide_names: Optional names attached to the individual style loss
            operator.

    Attributes:
        nst: NST image optimizer
    """

    def __init__(self, num_guides, impl_params=True, guide_names=None):
        nst = GatysEtAl2017SpatialControlNST(num_guides, impl_params, guide_names)
        super().__init__(nst, impl_params)
