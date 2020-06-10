Optimization
============

The merging of the identified content and style with a Neural Style Transfer (NST) is
posed as an optimization problem. The optimization is performed on the basis of a
:class:`~pystiche.loss.PerceptualLoss`. A distinction is made between two different
approaches.

Image optimization
------------------

In its basic form, an NST optimizes the pixels of the ``input_image`` directly. That
means they are iteratively adapted to reduce the perceptual loss. This
process is called *image optimization* and can be performed in ``pystiche`` with a
:func:`~pystiche.optim.default_image_optim_loop` .

Model optimization
------------------

While the image optimization approach yields the highest quality results, the
computation is quite expensive and usually takes multiple minutes to complete for a
single image. *Model optimization* on the other hand trains a model called
``transformer`` to perform the stylization. The training is performed with the same
perceptual loss as before, but now the ``transformer`` weights are used as optimization
parameters. The training is even more time consuming but afterwards the stylization is
performed in a single forward pass of the ``input_image`` through the ``transformer``.
The quality however, while still high, is lower than for image optimisation approaches
since the ``transformer`` cannot finetune the ``output_image``. In ``pystiche`` a model
optimization can be performed with a
:func:`~pystiche.optim.default_transformer_optim_loop` .

.. note::
  Due to the differences in execution time image and model optimization approaches are
  often dubbed *slow* and *fast* respectively.
