``pystiche.optim``
==================

.. automodule:: pystiche.optim


Optimization
------------

.. autofunction:: default_image_optimizer
.. autofunction:: image_optimization
.. autofunction:: pyramid_image_optimization

.. autofunction:: default_model_optimizer
.. autofunction:: model_optimization
.. autofunction:: multi_epoch_model_optimization


Logging
-------

.. autofunction:: default_logger

.. autoclass:: OptimLogger
  :members:
    message,
    sepline,
    sep_message,
    environment

.. autofunction:: default_image_optim_log_fn
.. autofunction:: default_pyramid_level_header

.. autofunction:: default_transformer_optim_log_fn
.. autofunction:: default_epoch_header
