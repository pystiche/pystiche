Installation
============

``pystiche`` is a proper Python package and can be installed with ``pip``. To install
the latest version run

.. code-block:: sh

  pip install git+https://github.com/pmeier/pystiche

.. note::

  ``pystiche`` is not yet listed on `PyPI <https://pypi.org/>`_, since it will be
  reviewed at `pyOpenSci <https://github.com/pmeier/pystiche/issues/93>`_ .


Installation of PyTorch
-----------------------

``pystiche`` is built upon `PyTorch <https://pytorch.org>`_ and depends on
``torch`` and ``torchvision``. By default they are installed with GPU support, which
significantly increases the download size as well as memory requirement during the
installation.

If you encounter issues during the installation or want to install without GPU support
please follow the
`official installation instructions of PyTorch <https://pytorch.org/get-started/>`_ for
your setup before you install ``pystiche``.

.. note::

  While ``pystiche`` is designed to be fully functional without a GPU, most tasks
  require significantly more time to perform on a CPU.
