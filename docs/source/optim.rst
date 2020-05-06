Optimization
============

The merging of the identified content and style with a Neural Style Transfer (NST) is
posed as an optimization problem. The optimization is performed on the basis of a
:class:`~pystiche.loss.perceptual.PerceptualLoss`. A
:class:`~pystiche.loss.perceptual.PerceptualLoss` combines multiple
:class:`~pystiche.ops.op.Operator` s into a joined ``criterion`` that measures how
well the ``input_image`` matches the content and style of the target images.

In its basic form, an NST optimizes the pixels of the ``input_image`` directly. That
means they are iteratively adapted to reduce the perceptual loss. This
process is called *image optimization* and can be performed in ``pystiche`` with an
:func:`~pystiche.optim.optim.default_image_optim_loop` .

While the image optimization approach yields the highest results, the computation is
quite expensive and usually takes multiple minutes to complete for a single image.
*Model optimization* on the other hand trains a network called ``transformer`` to
perform the stylization. The training is performed with the same perceptual loss as
before, but now the ``transformer`` weights are are used as optimization parameters.
The training is even more time consuming but afterwards the stylization is performed in
a single forward pass of the ``input_image`` through the ``transformer``. The quality
however, while still high, is lower than for image optimisation approaches since the
``transformer`` cannot finetune the ``output_image``. In ``pystiche`` a model
optimisation can be performed with an
:func:`~pystiche.optim.optim.default_transformer_optim_loop` .

.. note::
  Due to the execution time differences image and model optimization approaches are
  often dubbed *slow* and *fast* respectively.
