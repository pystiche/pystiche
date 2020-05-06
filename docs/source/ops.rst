Operators
=========

The identification of the content and style are core elements of a Neural Style
Transfer (NST). In ``pystiche`` this is performed by :class:`~pystiche.ops.op.Operator`
s. Every :class:`~pystiche.ops.op.Operator` is callable with an ``input_image`` and
returns a scalar ``score``. Depending on the specific implementation this returned
``score`` for example corresponds to how well the ``input_image`` matches the content
or style of a ``target_image``.

``pystiche`` differentiates between two :class:`~pystiche.ops.op.Operator` types:
:class:`~pystiche.ops.op.RegularizationOperator` and
:class:`~pystiche.ops.op.ComparisonOperator`. A
:class:`~pystiche.ops.op.RegularizationOperator` calculates the ``score`` of the
``ìnput_image`` without any context while a
:class:`~pystiche.ops.op.ComparisonOperator` compares the ``ìnput_image`` in some form
to a ``target_image``.

Furthermore, ``pystiche`` differentiates between two different domains an
:class:`~pystiche.ops.op.Operator` can work on: :class:`~pystiche.ops.op.PixelOperator`
and :class:`~pystiche.ops.op.EncodingOperator` . A
:class:`~pystiche.ops.op.PixelOperator` operates directly on the ``input_image`` while
a :class:`~pystiche.ops.op.EncodingOperator` encodes it first.

In total ``pystiche`` supports four arch types:

+----------------------------------------------------------+----------------------------------------------------------------------------------+
| :class:`~pystiche.ops.op.Operator`                       | Builtin examples                                                                 |
+==========================================================+==================================================================================+
| :class:`~pystiche.ops.op.PixelRegularizationOperator`    | - :class:`~pystiche.ops.regularization.TotalVariationOperator` :cite:`MV2014`    |
+----------------------------------------------------------+----------------------------------------------------------------------------------+
| :class:`~pystiche.ops.op.EncodingRegularizationOperator` |                                                                                  |
+----------------------------------------------------------+----------------------------------------------------------------------------------+
| :class:`~pystiche.ops.op.PixelComparisonOperator`        |                                                                                  |
+----------------------------------------------------------+----------------------------------------------------------------------------------+
| :class:`~pystiche.ops.op.EncodingComparisonOperator`     | - :class:`~pystiche.ops.comparison.FeatureReconstructionOperator` :cite:`MV2014` |
|                                                          | - :class:`~pystiche.ops.comparison.GramOperator` :cite:`GEB2016`                 |
|                                                          | - :class:`~pystiche.ops.comparison.MRFOperator` :cite:`LW2016`                   |
+----------------------------------------------------------+----------------------------------------------------------------------------------+