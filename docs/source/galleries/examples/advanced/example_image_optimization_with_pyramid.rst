.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_galleries_examples_advanced_example_image_optimization_with_pyramid.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_galleries_examples_advanced_example_image_optimization_with_pyramid.py:


Image-based optimization with image pyramid
===========================================

This example showcases how an image pyramid is integrated in an NST in ``pystiche`` .

With an image pyramid the optimization is not performed on single but rather on
multiple increasing resolutions. This procedure is often dubbed *coarse-to-fine*, since
on the lower resolutions coarse structures are synthesized whereas on the higher levels
the details are carved out.

This technique has the potential to reduce the convergence time as well as to enhance
the overall result :cite:`LW2016,GEB+2017`.

We start this example by importing everything we need and setting the device we will
be working on.


.. code-block:: default
   :lineno-start: 21


    import pystiche
    from pystiche.demo import demo_images
    from pystiche.enc import vgg19_multi_layer_encoder
    from pystiche.image import show_image, write_image
    from pystiche.loss import PerceptualLoss
    from pystiche.misc import get_device, get_input_image
    from pystiche.ops import (
        FeatureReconstructionOperator,
        MRFOperator,
        MultiLayerEncodingOperator,
    )
    from pystiche.optim import default_image_pyramid_optim_loop
    from pystiche.pyramid import ImagePyramid

    print(f"I'm working with pystiche=={pystiche.__version__}")

    device = get_device()
    print(f"I'm working with {device}")

    images = demo_images()
    images.download()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    I'm working with pystiche==0.4.0+dev.8c738e2.dirty
    I'm working with cuda




At first we define a :class:`~pystiche.loss.perceptual.PerceptualLoss` that is used
as optimization ``criterion``.


.. code-block:: default
   :lineno-start: 48


    multi_layer_encoder = vgg19_multi_layer_encoder()


    content_layer = "relu4_2"
    content_encoder = multi_layer_encoder.extract_single_layer_encoder(content_layer)
    content_weight = 1e0
    content_loss = FeatureReconstructionOperator(
        content_encoder, score_weight=content_weight
    )


    style_layers = ("relu3_1", "relu4_1")
    style_weight = 2e0


    def get_style_op(encoder, layer_weight):
        patch_size = 3
        return MRFOperator(encoder, patch_size, stride=2, score_weight=layer_weight)


    style_loss = MultiLayerEncodingOperator(
        multi_layer_encoder, style_layers, get_style_op, score_weight=style_weight,
    )

    criterion = PerceptualLoss(content_loss, style_loss).to(device)
    print(criterion)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    PerceptualLoss(
      (content_loss): FeatureReconstructionOperator(encoder=MultiLayerVGGEncoder(layer=relu4_2, arch=vgg19, weights=torch))
      (style_loss): MultiLayerEncodingOperator(
        encoder=MultiLayerVGGEncoder(arch=vgg19, weights=torch), score_weight=2
        (relu3_1): MRFOperator(score_weight=0.5, patch_size=(3, 3), stride=(2, 2))
        (relu4_1): MRFOperator(score_weight=0.5, patch_size=(3, 3), stride=(2, 2))
      )
    )




Opposed to the prior examples we want to perform an NST on multiple resolutions. In
``pystiche`` this handled by an :class:`~pystiche.pyramid.ImagePyramid` . The
resolutions are selected by specifying the ``edge_sizes`` of the images on each level
. The optimization is performed for ``num_steps`` on the different levels.

The resizing of all images, i.e. ``input_image`` and target images (``content_image``
and ``style_image``) is handled by the ``pyramid``. For that we need to register the
perceptual loss (``criterion``) as ``resize_targets``.

.. note::

  By default the ``edge_sizes`` correspond to the shorter ``edge`` of the images. To
  change that you can pass ``edge="long"``. For fine-grained control you can also
  pass a sequence comprising ``"short"`` and ``"long"`` to select the ``edge`` for
  each level separately.


.. code-block:: default
   :lineno-start: 93


    edge_sizes = (300, 550)
    num_steps = 200
    pyramid = ImagePyramid(edge_sizes, num_steps, resize_targets=(criterion,))
    print(pyramid)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ImagePyramid(
      (0): PyramidLevel(edge_size=300, num_steps=200, edge=short)
      (1): PyramidLevel(edge_size=550, num_steps=200, edge=short)
    )




Next up, we load and show the images that will be used in the NST.


.. code-block:: default
   :lineno-start: 102


    content_image = images["bird2"].read(device=device)
    show_image(content_image, title="Input image")





.. image:: /galleries/examples/advanced/images/sphx_glr_example_image_optimization_with_pyramid_001.png
    :class: sphx-glr-single-img






.. code-block:: default
   :lineno-start: 108


    style_image = images["mosaic"].read(device=device)
    show_image(style_image, title="Output image")





.. image:: /galleries/examples/advanced/images/sphx_glr_example_image_optimization_with_pyramid_002.png
    :class: sphx-glr-single-img





Although the images would be automatically resized during the optimization you might
need to resize them before: if you are working with large source images you might
run out of memory by setting up the targets of the perceptual loss. In that case it
is good practice to resize the images upfront to the largest size the ``pyramid``
will handle.


.. code-block:: default
   :lineno-start: 119


    top_level = pyramid[-1]
    content_image = top_level.resize_image(content_image)
    style_image = top_level.resize_image(style_image)









As a last preliminary step the previously loaded images are set as targets for the
perceptual loss (``criterion``) and we create the input image.


.. code-block:: default
   :lineno-start: 128


    criterion.set_content_image(content_image)
    criterion.set_style_image(style_image)

    starting_point = "content"
    input_image = get_input_image(starting_point, content_image=content_image)
    show_image(input_image, title="Input image")





.. image:: /galleries/examples/advanced/images/sphx_glr_example_image_optimization_with_pyramid_003.png
    :class: sphx-glr-single-img





Finally we run the NST with the
:func:`~pystiche.optim.optim.default_image_pyramid_optim_loop`. If ``get_optimizer``
is not specified, as is the case here, the
:func:`~pystiche.optim.optim.default_image_optimizer`, i.e.
:class:`~torch.optim.lbfgs.LBFGS` is used.


.. code-block:: default
   :lineno-start: 143


    output_image = default_image_pyramid_optim_loop(input_image, criterion, pyramid)

    show_image(output_image, title="Output image")
    write_image(output_image, "image_optimization_with_pyramid.jpg")



.. image:: /galleries/examples/advanced/images/sphx_glr_example_image_optimization_with_pyramid_004.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    |30.04.2020 15:59:41| ################################################################################
    |30.04.2020 15:59:41| Pyramid level 1 with 200 steps (378 x 300)
    |30.04.2020 15:59:41| ################################################################################
    |30.04.2020 15:59:43|   ================================================================================
    |30.04.2020 15:59:43|   Step 50
    |30.04.2020 15:59:43|   ================================================================================
    |30.04.2020 15:59:43|     content_loss: 7.210e+00
    |30.04.2020 15:59:43|     style_loss  : 1.743e+01
    |30.04.2020 15:59:45|   ================================================================================
    |30.04.2020 15:59:45|   Step 100
    |30.04.2020 15:59:45|   ================================================================================
    |30.04.2020 15:59:45|     content_loss: 7.319e+00
    |30.04.2020 15:59:45|     style_loss  : 1.592e+01
    |30.04.2020 15:59:48|   ================================================================================
    |30.04.2020 15:59:48|   Step 150
    |30.04.2020 15:59:48|   ================================================================================
    |30.04.2020 15:59:48|     content_loss: 7.310e+00
    |30.04.2020 15:59:48|     style_loss  : 1.539e+01
    |30.04.2020 15:59:51|   ================================================================================
    |30.04.2020 15:59:51|   Step 200
    |30.04.2020 15:59:51|   ================================================================================
    |30.04.2020 15:59:51|     content_loss: 7.303e+00
    |30.04.2020 15:59:51|     style_loss  : 1.511e+01
    |30.04.2020 15:59:51| ################################################################################
    |30.04.2020 15:59:51| Pyramid level 2 with 200 steps (693 x 550)
    |30.04.2020 15:59:51| ################################################################################
    |30.04.2020 15:59:57|   ================================================================================
    |30.04.2020 15:59:57|   Step 50
    |30.04.2020 15:59:57|   ================================================================================
    |30.04.2020 15:59:57|     content_loss: 5.864e+00
    |30.04.2020 15:59:57|     style_loss  : 2.195e+01
    |30.04.2020 16:00:04|   ================================================================================
    |30.04.2020 16:00:04|   Step 100
    |30.04.2020 16:00:04|   ================================================================================
    |30.04.2020 16:00:04|     content_loss: 5.823e+00
    |30.04.2020 16:00:04|     style_loss  : 2.110e+01
    |30.04.2020 16:00:11|   ================================================================================
    |30.04.2020 16:00:11|   Step 150
    |30.04.2020 16:00:11|   ================================================================================
    |30.04.2020 16:00:11|     content_loss: 5.804e+00
    |30.04.2020 16:00:11|     style_loss  : 2.077e+01
    |30.04.2020 16:00:18|   ================================================================================
    |30.04.2020 16:00:18|   Step 200
    |30.04.2020 16:00:18|   ================================================================================
    |30.04.2020 16:00:18|     content_loss: 5.794e+00
    |30.04.2020 16:00:18|     style_loss  : 2.058e+01





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  46.769 seconds)


.. _sphx_glr_download_galleries_examples_advanced_example_image_optimization_with_pyramid.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: example_image_optimization_with_pyramid.py <example_image_optimization_with_pyramid.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: example_image_optimization_with_pyramid.ipynb <example_image_optimization_with_pyramid.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
