from collections import OrderedDict
from torch import optim, nn
import pystiche
from pystiche.misc import to_engstr
from pystiche.encoding import vgg19_encoder
from pystiche.nst import (
    DirectEncodingComparisonOperator,
    GramEncodingComparisonOperator,
    MultiOperatorEncoder,
    CaffePreprocessingImageOptimizer,
)

__all__ = [
    "GatysEckerBethge2015ContentLoss",
    "GatysEckerBethge2015StyleLoss",
    "GatysEckerBethge2015NST",
]


def get_encoder():
    return vgg19_encoder(weights="caffe", preprocessing=False)


def optimizer_getter(input_image):
    return optim.LBFGS([input_image], lr=1.0, max_iter=1)


class GatysEckerBethge2015ContentLoss(DirectEncodingComparisonOperator):
    r"""Content loss from
    `"A Neural Algorithm of Artistic Style" <https://arxiv.org/abs/1508.06576>`

    Args:
        encoder: Encoder used to generate the feature maps. If None, a VGG19 encoder
            with weights from _Caffe_ and without internal preprocessing is used.
        impl_params: If True, hyper parameters from Johnsons implementation
            <https://github.com/jcjohnson/neural-style> rather than the parameters
            given in the paper are used.
    """

    def __init__(self, encoder=None, impl_params=True):
        if encoder is None:
            encoder = get_encoder()
        self.impl_params = impl_params

        name = "Content loss (direct)"
        layers = ("relu_4_2",)
        layer_weights = "sum" if impl_params else "mean"
        score_weight = 5e0 if impl_params else 1e0
        super().__init__(
            encoder,
            layers,
            name=name,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )

        self.score_correction_factor = 1.0 if impl_params else 1.0 / 2.0
        self.loss_reduction = "mean" if impl_params else "sum"

    def _calculate_score(self, *args, **kwargs):
        return super()._calculate_score(*args, **kwargs) * self.score_correction_factor

    def extra_descriptions(self):
        dct = OrderedDict()
        dct["Implementation parameters"] = self.impl_params
        if self.score_correction_factor != 1.0:
            dct["Score correction factor"] = to_engstr(self.score_correction_factor)
        if self.loss_reduction != "mean":
            dct["Loss reduction"] = self.loss_reduction
        return dct


class GatysEckerBethge2015StyleLoss(GramEncodingComparisonOperator):
    r"""Style loss based on Gram matrices from
    `"A Neural Algorithm of Artistic Style" <https://arxiv.org/abs/1508.06576>`

    Args:
        encoder: Encoder used to generate the feature maps. If None, a VGG19 encoder
            with weights from _Caffe_ and without internal preprocessing is used.
        impl_params: If True, hyper parameters from Johnsons implementation
            <https://github.com/jcjohnson/neural-style> rather than the parameters
            given in the paper are used.
    """

    def __init__(self, encoder=None, impl_params=False):
        if encoder is None:
            encoder = get_encoder()
        self.impl_params = impl_params

        name = "Style loss (Gram)"
        layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
        layer_weights = "sum" if impl_params else "mean"
        normalize = True
        score_weight = 1e2 if impl_params else 1e3
        super().__init__(
            encoder,
            layers,
            name=name,
            normalize=normalize,
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


def max_to_avg_pooling(nst_encoder):
    encoder = nst_encoder._encoder
    for name, module in encoder.named_children():
        if isinstance(module, nn.MaxPool2d):
            encoder._modules[name] = nn.AvgPool2d(**pystiche.pool_module_meta(module))


class GatysEckerBethge2015NST(CaffePreprocessingImageOptimizer):
    r"""NST from
    `"A Neural Algorithm of Artistic Style" <https://arxiv.org/abs/1508.06576>`

    A VGG19 encoder with weights from _Caffe_ and without internal preprocessing is
    used. The image is optimized with LBFGS.

    Args:
        impl_params: If True, hyper parameters from Johnsons implementation
            <https://github.com/jcjohnson/neural-style> rather than the parameters
            given in the paper are used.

    Attributes:
        content_loss: Content loss operator
        style_loss: Style loss operator
    """

    def __init__(self, impl_params=True):
        self.impl_params = impl_params

        encoder = MultiOperatorEncoder(get_encoder())
        if not impl_params:
            max_to_avg_pooling(encoder)

        content_loss = GatysEckerBethge2015ContentLoss(encoder)
        style_loss = GatysEckerBethge2015StyleLoss(encoder)

        super().__init__(content_loss, style_loss, optimizer_getter=optimizer_getter)
        self.content_loss = content_loss
        self.style_loss = style_loss
