Perceptual loss
===============

The identification of content and style are core elements of a Neural Style Transfer
(NST). The agreement of the content and style of two images is measured with the
``content_loss`` and ``style_loss``, respectively.


Operators
---------

In ``pystiche`` these losses are implemented :class:`~pystiche.loss.Loss` s.
:class:`~pystiche.loss.Loss` s are differentiated between two  types:
:class:`~pystiche.loss.RegularizationLoss` and
:class:`~pystiche.loss.ComparisonLoss`. A
:class:`~pystiche.loss.RegularizationLoss` works without any context while a
:class:`~pystiche.loss.ComparisonLoss` compares two images. Furthermore,
``pystiche`` differentiates between two different domains an
:class:`~pystiche.loss.Loss` can work on:
:class:`~pystiche.ops.op.PixelOperator`
and :class:`~pystiche.ops.EncodingOperator` . A :class:`~pystiche.ops.PixelOperator`
operates directly on the ``input_image`` while an
:class:`~pystiche.ops.EncodingOperator` encodes it first.

In total ``pystiche`` supports four archetypes:

+-------------------------------------------------------+-----------------------------------------------------------------------+
| :class:`~pystiche.loss.Loss`                          | Builtin examples                                                      |
+=======================================================+=======================================================================+
| :class:`~pystiche.ops.PixelRegularizationOperator`    |   - :class:`~pystiche.loss.TotalVariationLoss` :cite:`MV2015`         |
+-------------------------------------------------------+-----------------------------------------------------------------------+
| :class:`~pystiche.ops.EncodingRegularizationOperator` |                                                                       |
+-------------------------------------------------------+-----------------------------------------------------------------------+
| :class:`~pystiche.ops.PixelComparisonOperator`        |                                                                       |
+-------------------------------------------------------+-----------------------------------------------------------------------+
| :class:`~pystiche.ops.EncodingComparisonOperator`     | - :class:`~pystiche.loss.FeatureReconstructionLoss` :cite:`MV2015`    |
|                                                       | - :class:`~pystiche.loss.GramLoss` :cite:`GEB2016`                    |
|                                                       | - :class:`~pystiche.loss.MRFLoss` :cite:`LW2016`                      |
+-------------------------------------------------------+-----------------------------------------------------------------------+

Multi-layer encoder
-------------------

One of the main improvements of NST compared to traditional approaches is that the
agreement is not measured in the pixel or a handcrafted feature space, but rather in
the learned feature space of a Convolutional Neural Network called ``encoder``.
Especially variants of the ``style_loss`` depend upon encodings, i. e. feature maps,
from various layers of the encoder.

``pystiche`` offers a
:class:`~pystiche.enc.MultiLayerEncoder` that enables to extract all required encodings
after a single forward pass. If the same operator should be applied to different layers
of a :class:`~pystiche.enc.MultiLayerEncoder`, a
:class:`~pystiche.loss.MultiLayerEncodingLoss` can be used.


Perceptual loss
---------------

The :class:`~pystiche.loss.PerceptualLoss` combines all :class:`~pystiche.ops.Operator`
s in a single measure acting as joint optimization criterion. How the optimization is
performed will be detailed in the next section.
