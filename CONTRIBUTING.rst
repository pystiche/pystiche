Contributing guide lines
========================

We appreciate all contributions. If you are planning to contribute bug-fixes or
documentation improvements, please open a
`pull request (PR) <https://github.com/pmeier/pystiche/pulls>`_ without further
discussion. If you planning to contribute new features, please open an
`issue <https://github.com/pmeier/pystiche/issues>`_ and discuss the feature with us
first.

Every PR is subjected to multiple checks that it has to pass before it can be merged.
Below you can find details about these checks and instructions how to run them locally.

Lint
----

.. note::

  You can install all lint requirements with

  .. code-block:: sh

    pip install -r $PYSTICHE_ROOT/requirements-lint.txt

``pystiche`` uses `isort <https://timothycrosley.github.io/isort/>`_ to sort the
imports, `black <https://black.readthedocs.io/en/stable/>`_ to format the code, and
`flake8 <https://flake8.pycqa.org/en/latest/>`_ to enforce
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ compliance.

Furthermore, ``pystiche`` is `PEP561 <https://www.python.org/dev/peps/pep-0561/>`_
compliant and checks the type annotations with `mypy <http://mypy-lang.org/>`_ .

You can run all tools locally with:

.. code-block:: sh

  cd $PYSTICHE_ROOT
  isort --recursive .
  black .
  flake8
  mypy

.. note::

  The checks with ``isort``, ``black``, and ``flake8`` can be executed as a pre-commit
  hook. You can install them with:

  .. code-block:: sh

    pip install pre-commit
    cd $PYSTICHE_ROOT
    pre-commit install

  ``mypy`` is excluded from this, since the pre-commit runs in a separate virtual
  environment in which ``pystiche`` would have to be installed in for every commit.

Test
----

.. note::

  You can install all test requirements with

  .. code-block:: sh

    pip install -r $PYSTICHE_ROOT/requirements-test.txt

``pystiche`` uses `pytest <https://docs.pytest.org/en/stable/>`_ to run the test suite.
You can run it locally with:

.. code-block:: sh

  cd $PYSTICHE_HOME
  pytest


Documentation
-------------

.. note::

  You can install all documentation requirements with

  .. code-block:: sh

    pip install -r $PYSTICHE_ROOT/requirements-doc.txt

To build the documentation locally, run

.. code-block:: sh

  cd $PYSTICHE_ROOT/docs
  make $TARGET

You can run ``make`` without arguments to get a list of all available ``$TARGET`` s.
``TARGET=html`` and ``TARGET=latex`` are checked within a PR.

.. note::

  Running ``make $TARGET`` by default triggers a
  `sphinx gallery <https://sphinx-gallery.github.io/stable/index.html>`_ build for the
  examples , which will take some time to complete. To get around this, ``pystiche``
  offers two environment variables:

  - ``PYSTICHE_PLOT_GALLERY``: If ``False``, the code inside the galleries is not
    executed. See the
    `official sphinx-gallery documentation <https://sphinx-gallery.github.io/stable/configuration.html#without-execution>`_
    for details. Defaults to ``True``.
  - ``PYSTICHE_DOWNLOAD_GALLERY``: If ``True``, downloads the latest pre-built
    galleries before the built. Thus, during the built only changed galleries have to
    be rebuilt. The pre-built galleries are at most six hours old. Defaults to
    ``False``.

  Both environment variables are evaluated with :func:`~distutils.util.strtobool`.
