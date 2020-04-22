.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_tutorials_tutorial_image_based_optimization.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_tutorials_tutorial_image_based_optimization.py:


NST via image-based optimization
================================

imports


.. code-block:: default


    import torch
    from torch import optim
    from pystiche.image import show_image, write_image
    from pystiche.enc import vgg19_multi_layer_encoder
    from pystiche.ops import MSEEncodingOperator, GramOperator, MultiLayerEncodingOperator
    from pystiche.loss import PerceptualLoss
    from pystiche.demo import demo_images









Make this demo device-agnostic


.. code-block:: default


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")









Load the encoder used to create the feature maps for the NST


.. code-block:: default


    multi_layer_encoder = vgg19_multi_layer_encoder()









Create the content loss


.. code-block:: default


    content_layer = "relu4_2"
    content_encoder = multi_layer_encoder.extract_single_layer_encoder(content_layer)
    content_weight = 1e0
    content_loss = MSEEncodingOperator(content_encoder, score_weight=content_weight)









Create the style loss


.. code-block:: default


    style_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
    style_weight = 1e4


    def get_style_op(encoder, layer_weight):
        return GramOperator(encoder, score_weight=layer_weight)


    style_loss = MultiLayerEncodingOperator(
        multi_layer_encoder, style_layers, get_style_op, score_weight=style_weight,
    )









Combine the content and style loss into the optimization criterion


.. code-block:: default


    criterion = PerceptualLoss(content_loss, style_loss).to(device)
    print(criterion)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    PerceptualLoss(
      (content_loss): MSEEncodingOperator(encoder=MultiLayerVGGEncoder(layer=relu4_2, arch=vgg19, weights=torch))
      (style_loss): MultiLayerEncodingOperator(
        encoder=MultiLayerVGGEncoder(arch=vgg19, weights=torch), score_weight=10e3
        (relu1_1): GramOperator(score_weight=0.2)
        (relu2_1): GramOperator(score_weight=0.2)
        (relu3_1): GramOperator(score_weight=0.2)
        (relu4_1): GramOperator(score_weight=0.2)
        (relu5_1): GramOperator(score_weight=0.2)
      )
    )




load the content and style images and transfer them to the selected device
the images are resized, since the stylization is memory intensive


.. code-block:: default


    size = 500
    images = demo_images()
    content_image = images["dancing"].read(size=size, device=device)
    style_image = images["picasso"].read(size=size, device=device)
    show_image(content_image)
    show_image(style_image)





.. image:: /auto_tutorials/images/sphx_glr_tutorial_image_based_optimization_001.png
    :class: sphx-glr-single-img





Set the target images for the content and style loss


.. code-block:: default


    criterion.set_content_image(content_image)
    criterion.set_style_image(style_image)








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

Create the optimizer that performs the stylization


.. code-block:: default


    optimizer = optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)









.. note::
  To avoid boilerplate code, you can achieve the same behavior with
  :func:`~pystiche.optim.optim.default_image_optimizer`::

    from pystiche.optim import default_image_optimizer

    optimizer = default_image_optimizer(input_image)

Run the stylization


.. code-block:: default


    num_steps = 500
    for step in range(1, num_steps + 1):

        def closure():
            optimizer.zero_grad()
            loss = criterion(input_image)
            loss.backward()

            if step % 50 == 0:
                print(f"Step {step}")
                print()
                print(loss.aggregate(1))
                print("-" * 80)

            return loss

        optimizer.step(closure)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Step 50

    content_loss: 2.622e+00
    style_loss  : 8.435e+01
    --------------------------------------------------------------------------------
    Step 100

    content_loss: 2.700e+00
    style_loss  : 3.251e+01
    --------------------------------------------------------------------------------
    Step 150

    content_loss: 2.723e+00
    style_loss  : 1.752e+01
    --------------------------------------------------------------------------------
    Step 200

    content_loss: 2.730e+00
    style_loss  : 1.152e+01
    --------------------------------------------------------------------------------
    Step 250

    content_loss: 2.730e+00
    style_loss  : 8.900e+00
    --------------------------------------------------------------------------------
    Step 300

    content_loss: 2.726e+00
    style_loss  : 7.672e+00
    --------------------------------------------------------------------------------
    Step 350

    content_loss: 2.720e+00
    style_loss  : 6.954e+00
    --------------------------------------------------------------------------------
    Step 400

    content_loss: 2.714e+00
    style_loss  : 6.505e+00
    --------------------------------------------------------------------------------
    Step 450

    content_loss: 2.709e+00
    style_loss  : 6.176e+00
    --------------------------------------------------------------------------------
    Step 500

    content_loss: 2.702e+00
    style_loss  : 5.918e+00
    --------------------------------------------------------------------------------




.. note::
  To avoid boilerplate code, you can achieve the same behavior with
  :func:`~pystiche.optim.optim.default_image_optim_loop`::

    from pystiche.optim import default_image_optim_loop

    default_image_optim_loop(
        input_image, criterion, optimizer=optimizer, num_steps=num_steps
    )

  If you do not pass ``optimizer``
  :func:`~pystiche.optim.optim.default_image_optimizer` is used.

Show the stylization result


.. code-block:: default


    show_image(input_image)



.. image:: /auto_tutorials/images/sphx_glr_tutorial_image_based_optimization_002.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  55.395 seconds)


.. _sphx_glr_download_auto_tutorials_tutorial_image_based_optimization.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: tutorial_image_based_optimization.py <tutorial_image_based_optimization.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: tutorial_image_based_optimization.ipynb <tutorial_image_based_optimization.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
