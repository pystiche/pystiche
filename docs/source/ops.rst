Operator
========

The core components of a Neural Style Transfer (NST) are the ``content_loss`` and
``style_loss``. In ``pystiche`` they are implemented with
:class:`~pystiche.ops.op.Operator` s.

Every :class:`~pystiche.ops.op.Operator` is callable and given an ``input_image`` it
calculates a weighted scalar ``score`` representing the corresponding partial loss.

Each :class:`~pystiche.ops.op.Operator` is one of two types
(:class:`~pystiche.ops.op.RegularizationOperator` or
:class:`~pystiche.ops.op.ComparisonOperator`) and operates in one of two domains
(:class:`~pystiche.ops.op.PixelOperator` or :class:`~pystiche.ops.op.EncodingOperator`).
They are combined into four ``ABC`` s which each specific
:class:`~pystiche.ops.op.Operator` should subclass.

:class:`~pystiche.ops.op.RegularizationOperator` vs. :class:`~pystiche.ops.op.ComparisonOperator`
-------------------------------------------------------------------------------------------------

A :class:`~pystiche.ops.op.RegularizationOperator` calculates the ``score`` of the
``ìnput_image`` without any context. In contrast, a
:class:`~pystiche.ops.op.ComparisonOperator` compares the ``ìnput_image`` in some form
with ``target_image``.


:class:`~pystiche.ops.op.PixelOperator` vs. :class:`~pystiche.ops.op.EncodingOperator`
--------------------------------------------------------------------------------------

A :class:`~pystiche.ops.op.PixelOperator` performs all calculations directly on the
pixels. In contrast, a :class:`~pystiche.ops.op.EncodingOperator` at first encodes an
image with a given ``encoder`` and subsequently operates on these ``enc`` oding.


:class:`~pystiche.ops.op.PixelRegularizationOperator`
-----------------------------------------------------

.. image:: graphics/ops/PixelRegularizationOperator.png
  :alt: Block diagram of PixelRegularizationOperator


Notable subclasses:

- :class:`~pystiche.ops.regularization.TotalVariationOperator`

:class:`~pystiche.ops.op.EncodingRegularizationOperator`
--------------------------------------------------------

.. image:: graphics/ops/EncodingRegularizationOperator.png
  :alt: Block diagram of EncodingRegularizationOperator

:class:`~pystiche.ops.op.PixelComparisonOperator`
-------------------------------------------------

.. image:: graphics/ops/PixelComparisonOperator.png
  :alt: Block diagram of PixelComparisonOperator

:class:`~pystiche.ops.op.EncodingComparisonOperator`
----------------------------------------------------

.. image:: graphics/ops/EncodingComparisonOperator.png
  :alt: Block diagram of EncodingComparisonOperator

Notable subclasses:

- :class:`~pystiche.ops.comparison.MSEEncodingOperator`
- :class:`~pystiche.ops.comparison.GramOperator`
- :class:`~pystiche.ops.comparison.MRFOperator`
