``pystiche.optim``
==================

.. automodule:: pystiche.optim


Optimization
------------

.. autofunction:: default_image_optimizer
.. autofunction:: default_image_optim_loop
.. autofunction:: default_image_pyramid_optim_loop

.. autofunction:: default_transformer_optimizer
.. autofunction:: default_transformer_optim_loop
.. autofunction:: default_transformer_epoch_optim_loop


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
