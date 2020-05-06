Operators
=========

The identification of the content and style are core elements of a Neural Style
Transfer (NST). In ``pystiche`` this is performed by :class:`~pystiche.ops.Operator`
s. Every :class:`~pystiche.ops.Operator` is callable with an ``input_image`` and
returns a scalar ``score``. Depending on the specific implementation this returned
``score`` for example corresponds to how well the ``input_image`` matches the content
or style of a ``target_image``.

``pystiche`` differentiates between two :class:`~pystiche.ops.Operator` types:
:class:`~pystiche.ops.RegularizationOperator` and
:class:`~pystiche.ops.ComparisonOperator`. A
:class:`~pystiche.ops.RegularizationOperator` calculates the ``score`` of the
``ìnput_image`` without any context while a :class:`~pystiche.ops.ComparisonOperator`
compares the ``ìnput_image`` in some form to a ``target_image``.

Furthermore, ``pystiche`` differentiates between two different domains an
:class:`~pystiche.ops.Operator` can work on: :class:`~pystiche.ops.op.PixelOperator`
and :class:`~pystiche.ops.EncodingOperator` . A :class:`~pystiche.ops.PixelOperator`
operates directly on the ``input_image`` while a
:class:`~pystiche.ops.EncodingOperator` encodes it first.

In total ``pystiche`` supports four arch types:

+-------------------------------------------------------+-----------------------------------------------------------------------+
| :class:`~pystiche.ops.Operator`                       | Builtin examples                                                      |
+=======================================================+=======================================================================+
| :class:`~pystiche.ops.PixelRegularizationOperator`    | - :class:`~pystiche.ops.TotalVariationOperator` :cite:`MV2014`        |
+-------------------------------------------------------+-----------------------------------------------------------------------+
| :class:`~pystiche.ops.EncodingRegularizationOperator` |                                                                       |
+-------------------------------------------------------+-----------------------------------------------------------------------+
| :class:`~pystiche.ops.PixelComparisonOperator`        |                                                                       |
+-------------------------------------------------------+-----------------------------------------------------------------------+
| :class:`~pystiche.ops.EncodingComparisonOperator`     | - :class:`~pystiche.ops.FeatureReconstructionOperator` :cite:`MV2014` |
|                                                       | - :class:`~pystiche.ops.GramOperator` :cite:`GEB2016`                 |
|                                                       | - :class:`~pystiche.ops.MRFOperator` :cite:`LW2016`                   |
+-------------------------------------------------------+-----------------------------------------------------------------------+