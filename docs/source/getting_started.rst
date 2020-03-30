Installation
============

``pystiche`` is a proper Python package and can be installed with ``pip``. To install
the latest version from source run

.. code-block:: sh

  git clone https://github.com/pmeier/pystiche
  cd pystiche
  pip install .

.. note::

  ``pystiche`` is not yet listed on `PyPI <https://pypi.org/>`_, since it is under
  review at `pyOpenSci <https://github.com/pmeier/pystiche/issues/93>`_ . It is
  recommended to
  `wait until the review process is finished <https://www.pyopensci.org/dev_guide/peer_review/author_guide.html#Packaging-Guide>`_
  before releasing on PyPI.

Installation without GPU
------------------------

``pystiche`` depends on ``torch`` and ``torchvision``. By default they are installed
with GPU support, which significantly increases the download size as well as memory
requirement during the installation. If you do not have access to a GPU or want to
install without GPU support for another reason, please follow the
`official installation instructions <https://pytorch.org/get-started/locally/>`_ .

.. note::

  While ``pystiche`` is designed to be fully functional without a GPU, most tasks
  that ``pystiche`` was built for require significantly more time to perform on a CPU.


Installation with Extras
------------------------

You can install ``pystiche`` with ``pip install .[$EXTRA]`` where ``$EXTRA`` is one of
the following:

- ``test``: Installs everything needed to run the test suite.
- ``doc``: Installs everything needed to build the documentation.
- ``dev``: Installs everything needed to work on ``pystiche``. This includes all
  dependencies from ``test`` and ``doc``.


Installation for developers
---------------------------

If you want to contribute to ``pystiche`` please install with the ``[dev]`` extra in
order to install all required development tools. Since ``pystiche`` uses the
`black code formatter <https://github.com/psf/black>`_, you should install it as a
pre-commit hook:

.. code-block:: sh

  pre-commit install


Documentation
-------------

The documentation for ``pystiche`` is hosted on
`Read the Docs <https://pystiche.readthedocs.io/en/latest/>`_ . If you want to build
the documentation locally run

.. code-block:: sh

  cd $PYSTICHE_ROOT/docs
  make $TARGET

You can run ``make`` without arguments to get a list of all available ``$TARGET``s.

.. note::

  Running ``make $TARGET`` by default triggers a
  `sphinx gallery <https://sphinx-gallery.github.io/stable/index.html>`_ build, which
  will take some time to complete. To exclude all galleries set the environment
  variable ``PYSTICHE_EXCLUDE_GALLERY`` to anything that evaluates to ``True``, i.e.
  ``export PYSTICHE_EXCLUDE_GALLERY=True``.