``pystiche.enc``
================

.. automodule:: pystiche.enc

.. autoclass:: Encoder
  :members:
    forward,
    propagate_guide
.. autoclass:: SequentialEncoder

.. autoclass:: MultiLayerEncoder
  :members:
    __contains__,
    forward,
    extract_encoder,
    encode,
    empty_storage,
    trim,
    propagate_guide
.. autoclass:: SingleLayerEncoder
  :members:
    forward,
    propagate_guide

Models
------

.. autoclass:: ModelMultiLayerEncoder
  :members:
    state_dict_url,
    collect_modules,
    load_state_dict,
    load_state_dict_from_url

VGG
^^^

.. autoclass:: VGGMultiLayerEncoder
.. autofunction:: vgg11_multi_layer_encoder
.. autofunction:: vgg11_bn_multi_layer_encoder
.. autofunction:: vgg13_multi_layer_encoder
.. autofunction:: vgg13_bn_multi_layer_encoder
.. autofunction:: vgg16_multi_layer_encoder
.. autofunction:: vgg16_bn_multi_layer_encoder
.. autofunction:: vgg19_multi_layer_encoder
.. autofunction:: vgg19_bn_multi_layer_encoder

AlexNet
^^^^^^^

.. autoclass:: AlexNetMultiLayerEncoder
.. autofunction:: alexnet_multi_layer_encoder
