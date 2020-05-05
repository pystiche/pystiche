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
`official installation instructions <https://pytorch.org/get-started/locally/>`_ for
your setup.

.. note::

  While ``pystiche`` is designed to be fully functional without a GPU, most tasks
  that ``pystiche`` was built for require significantly more time to perform on a CPU.


Installation with extras
------------------------

You can install ``pystiche`` with ``pip install .[$EXTRA]`` where ``$EXTRA`` is one of
the following:

- ``test``: Installs everything needed to run the test suite.
- ``doc``: Installs everything needed to build the documentation.
- ``dev``: Installs everything needed to work on ``pystiche``. This includes all
  dependencies from ``test`` and ``doc``.


Installation for developers
---------------------------

If you want to contribute to ``pystiche`` please install with the ``dev`` extra in
order to install all required development tools. For an automatic code format check you
can install the necessary ``pre-commit`` hooks with

.. code-block:: sh

  cd $PYSTICHE_ROOT/
  pre-commit install


Build documentation
-------------------

The documentation for ``pystiche`` is hosted on
`Read the Docs <https://pystiche.readthedocs.io/en/latest/>`_ . If you want to build
the documentation locally install with the ``doc`` extra and run

.. code-block:: sh

  cd $PYSTICHE_ROOT/docs
  make $TARGET

You can run ``make`` without arguments to get a list of all available ``$TARGET`` s.

.. note::

  Running ``make $TARGET`` by default triggers a
  `sphinx gallery <https://sphinx-gallery.github.io/stable/index.html>`_ build, which
  will take some time to complete. To exlucde this from the build set the environment
  variable ``PYSTICHE_PLOT_GALLERY`` to anything that evaluates to ``False`` with FIXME.