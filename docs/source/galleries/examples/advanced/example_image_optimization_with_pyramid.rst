.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_galleries_examples_advanced_example_image_optimization_with_pyramid.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_galleries_examples_advanced_example_image_optimization_with_pyramid.py:


Image-based optimization with image pyramid
===========================================

We start this example by importing everything we need and setting the device we will
be working on.


.. code-block:: default
   :lineno-start: 11


    import pystiche
    from pystiche.demo import demo_images
    from pystiche.enc import vgg19_multi_layer_encoder
    from pystiche.image import show_image, write_image
    from pystiche.loss import PerceptualLoss
    from pystiche.misc import get_device, get_input_image
    from pystiche.ops import (
        FeatureReconstructionOperator,
        GramOperator,
        MultiLayerEncodingOperator,
    )
    from pystiche.optim import default_image_pyramid_optim_loop
    from pystiche.pyramid import ImagePyramid

    print(f"I'm working with pystiche=={pystiche.__version__}")

    device = get_device()
    print(f"I'm working with {device}")

    images = demo_images()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    I'm working with pystiche==0.4.0+dev.a9b566d.dirty
    I'm working with cuda




At first we define a :class:`~pystiche.loss.perceptual.PerceptualLoss` that is used
as optimization ``criterion``.


.. code-block:: default
   :lineno-start: 37


    multi_layer_encoder = vgg19_multi_layer_encoder()


    content_layer = "relu4_2"
    content_encoder = multi_layer_encoder.extract_single_layer_encoder(content_layer)
    content_weight = 1e0
    content_loss = FeatureReconstructionOperator(
        content_encoder, score_weight=content_weight
    )


    style_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
    style_weight = 1e4


    def get_style_op(encoder, layer_weight):
        return GramOperator(encoder, score_weight=layer_weight)


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
        encoder=MultiLayerVGGEncoder(arch=vgg19, weights=torch), score_weight=10e3
        (relu1_1): GramOperator(score_weight=0.2)
        (relu2_1): GramOperator(score_weight=0.2)
        (relu3_1): GramOperator(score_weight=0.2)
        (relu4_1): GramOperator(score_weight=0.2)
        (relu5_1): GramOperator(score_weight=0.2)
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
   :lineno-start: 81


    edge_sizes = (500, 800)
    num_steps = (500, 200)
    pyramid = ImagePyramid(edge_sizes, num_steps, resize_targets=(criterion,))
    print(pyramid)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ImagePyramid(
      (0): PyramidLevel(edge_size=500, num_steps=500, edge=short)
      (1): PyramidLevel(edge_size=800, num_steps=200, edge=short)
    )




Next up, we load and show the images that will be used in the NST.


.. code-block:: default
   :lineno-start: 90


    content_image = images["dancing"].read(device=device)
    show_image(content_image, title="Input image")





.. image:: /galleries/examples/advanced/images/sphx_glr_example_image_optimization_with_pyramid_001.png
    :class: sphx-glr-single-img






.. code-block:: default
   :lineno-start: 96


    style_image = images["picasso"].read(device=device)
    show_image(style_image, title="Output image")





.. image:: /galleries/examples/advanced/images/sphx_glr_example_image_optimization_with_pyramid_002.png
    :class: sphx-glr-single-img





.. note::

  Although the images will be automatically resized during the optimization you might
  need to resize them before: if you are working with large source images you might
  run out of memory by setting up the targets of the perceptual loss. In that case it
  is good practice to resize the images upfront to the largest size the ``pyramid``
  will handle:

  .. code-block::

      top_level = pyramid[-1]
      image = top_level.resize(image)

As a last preliminary step the previously loaded images are set as targets for the
perceptual loss (``criterion``) and we create the input image.


.. code-block:: default
   :lineno-start: 119


    criterion.set_content_image(content_image)
    criterion.set_style_image(style_image)

    starting_point = "content"
    input_image = get_input_image(starting_point, content_image=content_image)
    show_image(input_image, title="Input image")





.. image:: /galleries/examples/advanced/images/sphx_glr_example_image_optimization_with_pyramid_003.png
    :class: sphx-glr-single-img





Finally we run the NST and afterwards show the result and save it.


.. code-block:: default
   :lineno-start: 130


    output_image = default_image_pyramid_optim_loop(input_image, criterion, pyramid)

    show_image(output_image, title="Output image")
    write_image(output_image, "image_optimization_with_pyramid.jpg")



.. image:: /galleries/examples/advanced/images/sphx_glr_example_image_optimization_with_pyramid_004.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    |24.04.2020 16:56:14| ################################################################################
    |24.04.2020 16:56:14| Pyramid level 1 with 500 steps (500 x 500)
    |24.04.2020 16:56:14| ################################################################################
    |24.04.2020 16:56:17|   ================================================================================
    |24.04.2020 16:56:17|   Step 50
    |24.04.2020 16:56:17|   ================================================================================
    |24.04.2020 16:56:17|     content_loss: 2.584e+00
    |24.04.2020 16:56:17|     style_loss  : 8.638e+01
    |24.04.2020 16:56:22|   ================================================================================
    |24.04.2020 16:56:22|   Step 100
    |24.04.2020 16:56:22|   ================================================================================
    |24.04.2020 16:56:22|     content_loss: 2.667e+00
    |24.04.2020 16:56:22|     style_loss  : 3.394e+01
    |24.04.2020 16:56:27|   ================================================================================
    |24.04.2020 16:56:27|   Step 150
    |24.04.2020 16:56:27|   ================================================================================
    |24.04.2020 16:56:27|     content_loss: 2.684e+00
    |24.04.2020 16:56:27|     style_loss  : 1.836e+01
    |24.04.2020 16:56:32|   ================================================================================
    |24.04.2020 16:56:32|   Step 200
    |24.04.2020 16:56:32|   ================================================================================
    |24.04.2020 16:56:32|     content_loss: 2.694e+00
    |24.04.2020 16:56:32|     style_loss  : 1.195e+01
    |24.04.2020 16:56:36|   ================================================================================
    |24.04.2020 16:56:36|   Step 250
    |24.04.2020 16:56:36|   ================================================================================
    |24.04.2020 16:56:36|     content_loss: 2.695e+00
    |24.04.2020 16:56:36|     style_loss  : 9.172e+00
    |24.04.2020 16:56:42|   ================================================================================
    |24.04.2020 16:56:42|   Step 300
    |24.04.2020 16:56:42|   ================================================================================
    |24.04.2020 16:56:42|     content_loss: 2.689e+00
    |24.04.2020 16:56:42|     style_loss  : 7.808e+00
    |24.04.2020 16:56:47|   ================================================================================
    |24.04.2020 16:56:47|   Step 350
    |24.04.2020 16:56:47|   ================================================================================
    |24.04.2020 16:56:47|     content_loss: 2.682e+00
    |24.04.2020 16:56:47|     style_loss  : 7.060e+00
    |24.04.2020 16:56:51|   ================================================================================
    |24.04.2020 16:56:51|   Step 400
    |24.04.2020 16:56:51|   ================================================================================
    |24.04.2020 16:56:51|     content_loss: 2.675e+00
    |24.04.2020 16:56:51|     style_loss  : 6.600e+00
    |24.04.2020 16:56:57|   ================================================================================
    |24.04.2020 16:56:57|   Step 450
    |24.04.2020 16:56:57|   ================================================================================
    |24.04.2020 16:56:57|     content_loss: 2.672e+00
    |24.04.2020 16:56:57|     style_loss  : 6.270e+00
    |24.04.2020 16:57:02|   ================================================================================
    |24.04.2020 16:57:02|   Step 500
    |24.04.2020 16:57:02|   ================================================================================
    |24.04.2020 16:57:02|     content_loss: 2.668e+00
    |24.04.2020 16:57:02|     style_loss  : 6.025e+00
    |24.04.2020 16:57:02| ################################################################################
    |24.04.2020 16:57:02| Pyramid level 2 with 200 steps (800 x 800)
    |24.04.2020 16:57:02| ################################################################################
    |24.04.2020 16:57:12|   ================================================================================
    |24.04.2020 16:57:12|   Step 50
    |24.04.2020 16:57:12|   ================================================================================
    |24.04.2020 16:57:12|     content_loss: 1.686e+00
    |24.04.2020 16:57:12|     style_loss  : 2.999e+00
    |24.04.2020 16:57:22|   ================================================================================
    |24.04.2020 16:57:22|   Step 100
    |24.04.2020 16:57:22|   ================================================================================
    |24.04.2020 16:57:22|     content_loss: 1.598e+00
    |24.04.2020 16:57:22|     style_loss  : 1.731e+00
    |24.04.2020 16:57:33|   ================================================================================
    |24.04.2020 16:57:33|   Step 150
    |24.04.2020 16:57:33|   ================================================================================
    |24.04.2020 16:57:33|     content_loss: 1.535e+00
    |24.04.2020 16:57:33|     style_loss  : 1.326e+00
    |24.04.2020 16:57:45|   ================================================================================
    |24.04.2020 16:57:45|   Step 200
    |24.04.2020 16:57:45|   ================================================================================
    |24.04.2020 16:57:45|     content_loss: 1.488e+00
    |24.04.2020 16:57:45|     style_loss  : 1.126e+00





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  37.095 seconds)


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
