Operator
========

The core components of a Neural Style Transfer (NST) are the ``content_loss`` and
``style_loss``. In ``pystiche`` they are implemented with
:class:`~pystiche.ops.Operator` s.

Every :class:`~pystiche.ops.Operator` is callable and given an ``input_image`` it
calculates a weighted scalar ``score`` representing the corresponding partial loss.

Each :class:`~pystiche.ops.Operator` is one of two types
(:class:`~pystiche.ops.RegularizationOperator` or
:class:`~pystiche.ops.ComparisonOperator`) and operates in one of two domains
(:class:`~pystiche.ops.PixelOperator` or :class:`~pystiche.ops.EncodingOperator`). They
are combined into four ABCs which each specific :class:`~pystiche.ops.Operator` should
subclass.


:class:`~pystiche.ops.RegularizationOperator` vs. :class:`~pystiche.ops.ComparisonOperator`
-------------------------------------------------------------------------------------------

- regularization operates on the image without context
    - usually for regulrarizers
- comparison compares the input image to some target
    - usually for content and style loss

:class:`~pystiche.ops.PixelOperator` vs. :class:`~pystiche.ops.EncodingOperator`
--------------------------------------------------------------------------------

- pixel operates directly on the image (regularizer)
- encoding operates on encodings of the image (content and style loss)

:class:`~pystiche.ops.PixelRegularizationOperator`
--------------------------------------------------

:class:`~pystiche.ops.EncodingRegularizationOperator`
-----------------------------------------------------

:class:`~pystiche.ops.PixelComparisonOperator`
----------------------------------------------

:class:`~pystiche.ops.EncodingComparisonOperator`
-------------------------------------------------
