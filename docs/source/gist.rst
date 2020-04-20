Gist
====

In general, a Neural StyleTransfer (NST) merges the content of one image and the style of another into a new stylized image.

### IMAGE

This process is controlled by optimizing a multi-goal criterion called *perceptual loss*. This perceptual loss usually comprises a ``content_loss`` and a ``style_loss`` and optionally a ``regularizer``. As the name implies the ``content``- and ``style_loss`` indicate how well the stylized image matches the content and style of the target images. A regularizer can be used to suppress artifacts in the stylized image.

The following sections provide the gist of how a NST is performed with ``pystiche``.

.. toctree::
  :maxdepth: 2

  Operators <ops>
  Optimization <optim>
