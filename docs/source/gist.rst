.. _gist:

Gist
====

From a high viewpoint, Neural Style Transfer (NST) can be described with only three
images and two symbols:

.. image:: graphics/banner/banner.jpg
    :alt: pystiche banner

Not only the quality of the results but also the underlying steps are comparable to the
work of human artisans or craftsmen. Such a manual style transfer can be roughly
divided into three steps:

1. The content or motif of an image needs to be identified. That means one has to
   identify which parts of the image are essential and on the other hand which details
   can be discarded.
2. The style of an image, such as color, shapes, brush strokes, needs to be identified.
   Usually that means one has to intensively study of the works of the original artist.
3. The identified content and style have to be merged together. This can be the most
   difficult step, since it usually requires a lot of skill to match the style of
   another artist.

In principle an NST performs the same steps, albeit fully automatically. This is
nothing new in the field of computational style transfers. What makes NST stand out is
its generality: NST only needs a single arbitrary content and style image as input and
thus "makes -- for the first time -- a generalized style transfer practicable."
:cite:`SID2017`.

The following sections provide the gist of how these three steps are performed with
``pystiche`` as part of an NST . Afterwards head over to the
:ref:`usage examples <usage_examples>` to see ``pystiche`` in action.

.. toctree::
  :maxdepth: 2

  Perceptual loss <loss>
  Optimization <optim>
