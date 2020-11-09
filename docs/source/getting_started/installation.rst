Installation
============

The latest **stable** version can be installed with

.. code-block:: sh

  pip install pystiche

The latest **potentially unstable** version can be installed with

.. code-block::

  pip install git+https://github.com/pmeier/pystiche@master


Installation of PyTorch
-----------------------

``pystiche`` is built upon `PyTorch <https://pytorch.org>`_ and depends on
``torch`` and ``torchvision``. By default, a ``pip install`` of ``pystiche`` tries to
install the PyTorch distributions precompiled for the latest CUDA release. If you use
another version or don't have a CUDA-capable GPU, we encourage you to try
`light-the-torch <https://github.com/pmeier/light-the-torch>`_ for a convenient
installation:

.. code-block:: sh

  pip install light-the-torch
  ltt install pystiche

Otherwise, please follow the
`official installation instructions of PyTorch <https://pytorch.org/get-started/>`_ for
your setup before you install ``pystiche``.

.. note::

  While ``pystiche`` is designed to be fully functional without a GPU, most tasks
  require significantly more time to perform on a CPU.
