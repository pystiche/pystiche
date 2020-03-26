.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_tutorials_tutorial_image_based_optimization_with_pyramid.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_tutorials_tutorial_image_based_optimization_with_pyramid.py:


NST via image-based optimization with image pyramid
===================================================

imports


.. code-block:: default


    from collections import OrderedDict
    import torch
    from torch import optim
    from pystiche.image import extract_aspect_ratio, show_image, write_image
    from pystiche.enc import vgg19_encoder
    from pystiche.ops import MSEEncodingOperator, GramOperator, MultiLayerEncodingOperator
    from pystiche.loss import MultiOperatorLoss
    from pystiche.pyramid import ImagePyramid
    from pystiche.demo import demo_images









Make this demo device-agnostic


.. code-block:: default


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")









Load the encoder used to create the feature maps for the NST


.. code-block:: default


    multi_layer_encoder = vgg19_encoder()









Create the content loss


.. code-block:: default


    content_layer = "relu_4_2"
    content_encoder = multi_layer_encoder[content_layer]
    content_weight = 1e0
    content_loss = MSEEncodingOperator(content_encoder, score_weight=content_weight)









Create the style loss


.. code-block:: default


    style_layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
    style_weight = 1e4


    def get_style_op(encoder, layer_weight):
        return GramOperator(encoder, score_weight=layer_weight)


    style_loss = MultiLayerEncodingOperator(
        multi_layer_encoder, style_layers, get_style_op, score_weight=style_weight,
    )









Combine the content and style loss into the optimization criterion


.. code-block:: default


    criterion = MultiOperatorLoss(
        OrderedDict([("content_loss", content_loss), ("style_loss", style_loss)])
    )
    criterion = criterion.to(device)









Create the image pyramid used for the stylization


.. code-block:: default


    edge_sizes = (500, 700)
    num_steps = (500, 200)
    pyramid = ImagePyramid(edge_sizes, num_steps, resize_targets=(criterion,))









load the content and style images and transfer them to the selected device


.. code-block:: default


    images = demo_images()
    content_image = images["dancing"].read(device=device)
    style_image = images["picasso"].read(device=device)









resize the images, since the stylization is memory intensive


.. code-block:: default


    resize = pyramid[-1].resize_image
    content_image = resize(content_image)
    style_image = resize(style_image)
    show_image(content_image)
    show_image(style_image)





.. image:: /auto_tutorials/images/sphx_glr_tutorial_image_based_optimization_with_pyramid_001.png
    :class: sphx-glr-single-img





Set the target images for the content and style loss


.. code-block:: default


    content_loss.set_target_image(content_image)
    style_loss.set_target_image(style_image)









Set the starting point of the stylization to the content image. If you want
to start from a white noise image instead, uncomment the line below


.. code-block:: default


    input_image = content_image.clone()









.. note::
  To avoid boilerplate code, you can achieve the same behavior with
  :func:`~pystiche.misc.misc.get_input_image`::

    from pystiche.misc import get_input_image

    starting_point = "content"
    input_image = get_input_image(starting_point, content_image=content_image)

.. note::
  If you want to start the stylization from a white noise image instead, you
  can use::

    input_image = torch.rand_like(content_image)

  or::

    starting_point = "random"
    input_image = get_input_image(starting_point, content_image=content_image)

extract the original aspect ratio to avoid size mismatch errors during resizing


.. code-block:: default


    aspect_ratio = extract_aspect_ratio(input_image)









Define a getter for the optimizer that performs the stylization


.. code-block:: default



    def get_optimizer(input_image):
        return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)









Run the stylization


.. code-block:: default


    for num_level, level in enumerate(pyramid, 1):
        input_image = level.resize_image(input_image, aspect_ratio=aspect_ratio)
        optimizer = get_optimizer(input_image)

        for step in level:

            def closure():
                optimizer.zero_grad()
                loss = criterion(input_image)
                loss.backward()

                if step % 50 == 0:
                    print(f"Level {num_level}, Step {step}")
                    print()
                    print(loss.aggregate(1))
                    print("-" * 80)

                return loss

            optimizer.step(closure)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Level 1, Step 50

    content_loss: 2.473e+00
    style_loss  : 8.308e+01
    --------------------------------------------------------------------------------
    Level 1, Step 100

    content_loss: 2.559e+00
    style_loss  : 3.479e+01
    --------------------------------------------------------------------------------
    Level 1, Step 150

    content_loss: 2.585e+00
    style_loss  : 1.921e+01
    --------------------------------------------------------------------------------
    Level 1, Step 200

    content_loss: 2.590e+00
    style_loss  : 1.228e+01
    --------------------------------------------------------------------------------
    Level 1, Step 250

    content_loss: 2.592e+00
    style_loss  : 9.073e+00
    --------------------------------------------------------------------------------
    Level 1, Step 300

    content_loss: 2.592e+00
    style_loss  : 7.637e+00
    --------------------------------------------------------------------------------
    Level 1, Step 350

    content_loss: 2.587e+00
    style_loss  : 6.852e+00
    --------------------------------------------------------------------------------
    Level 1, Step 400

    content_loss: 2.582e+00
    style_loss  : 6.361e+00
    --------------------------------------------------------------------------------
    Level 1, Step 450

    content_loss: 2.579e+00
    style_loss  : 6.014e+00
    --------------------------------------------------------------------------------
    Level 1, Step 500

    content_loss: 2.574e+00
    style_loss  : 5.758e+00
    --------------------------------------------------------------------------------
    Level 2, Step 50

    content_loss: 1.920e+00
    style_loss  : 4.788e+00
    --------------------------------------------------------------------------------
    Level 2, Step 100

    content_loss: 1.849e+00
    style_loss  : 2.728e+00
    --------------------------------------------------------------------------------
    Level 2, Step 150

    content_loss: 1.800e+00
    style_loss  : 2.091e+00
    --------------------------------------------------------------------------------
    Level 2, Step 200

    content_loss: 1.764e+00
    style_loss  : 1.785e+00
    --------------------------------------------------------------------------------




.. note::
  To avoid boilerplate code, you can achieve the same behavior with
  :func:`~pystiche.optim.optim.default_image_pyramid_optim_loop`::

    from pystiche.optim import default_image_pyramid_optim_loop

    input_image = default_image_pyramid_optim_loop(
        input_image, criterion, pyramid, get_optimizer=get_optimizer
    )

  If you do not pass ``get_optimizer``
  :func:`~pystiche.optim.optim.default_image_optimizer` is used.

Show the stylization result


.. code-block:: default


    show_image(input_image)



.. image:: /auto_tutorials/images/sphx_glr_tutorial_image_based_optimization_with_pyramid_002.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  24.973 seconds)


.. _sphx_glr_download_auto_tutorials_tutorial_image_based_optimization_with_pyramid.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: tutorial_image_based_optimization_with_pyramid.py <tutorial_image_based_optimization_with_pyramid.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: tutorial_image_based_optimization_with_pyramid.ipynb <tutorial_image_based_optimization_with_pyramid.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
