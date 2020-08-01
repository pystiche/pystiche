Contributing guide lines
========================

We appreciate all contributions. If you are planning to contribute bug-fixes or
documentation improvements, please open a
`pull request (PR) <https://github.com/pmeier/pystiche/pulls>`_ without further
discussion. If you planning to contribute new features, please open an
`issue <https://github.com/pmeier/pystiche/issues>`_ and discuss the feature with us
first.

To start working on ``pystiche`` clone from the latest version and install the
development requirements:

.. code-block:: sh

  PYSTICHE_ROOT = pystiche_papers
  git clone https://github.com/pmeier/pystiche $PYSTICHE_ROOT
  cd $PYSTICHE_ROOT
  pip install -r requirements-dev.txt
  pre-commit install

Every PR is subjected to multiple checks that it has to pass before it can be merged.
The checks are performed by `tox <https://tox.readthedocs.io/en/latest/>`_ . Below
you can find details and instructions how to run the checks locally.

Code format and linting
-----------------------

``pystiche`` uses `isort <https://timothycrosley.github.io/isort/>`_ to sort the
imports, `black <https://black.readthedocs.io/en/stable/>`_ to format the code, and
`flake8 <https://flake8.pycqa.org/en/latest/>`_ to enforce
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ compliance. To format and check the
code style, run

.. code-block:: sh

  cd $PYSTICHE_ROOT
  tox -e lint-style

.. note::

  Amongst others, ``isort``, ``black``, and ``flake8`` are run by
  `pre-commit <https://pre-commit.com/>`_ before every commit.

Furthermore, ``pystiche_papers`` is
`PEP561 <https://www.python.org/dev/peps/pep-0561/>`_ compliant and checks the type
annotations with `mypy <http://mypy-lang.org/>`_ . To check the static typing, run

.. code-block:: sh

  cd $PYSTICHE_ROOT
  tox -e lint-typing

You can run all lint checks with

.. code-block:: sh

  cd $PYSTICHE_ROOT
  tox -f lint


Tests
-----

``pystiche`` uses `pytest <https://docs.pytest.org/en/stable/>`_ to run the test suite.
You can run it locally with

.. code-block:: sh

  cd $PYSTICHE_ROOT
  tox

.. note::

  ``pystiche_papers`` adds the following custom options with the
  corresponding ``@pytest.mark.*`` decorators:
  - ``--skip-large-download``: ``@pytest.mark.large_download``
  - ``--skip-slow``: ``@pytest.mark.slow``
  - ``--run-flaky``: ``@pytest.mark.flaky``

  Options prefixed with ``--skip`` are run by default and skipped if the option is
  given. Options prefixed with ``--run`` are skipped by default and run if the option
  is given.

  These options are passed through ``tox`` if given after a ``--`` flag. For example,
  the CI invokes the test suite with

  .. code-block:: sh

    cd $PYSTICHE_ROOT
    tox -- --skip-large-download


Documentation
-------------

To build the html documentation locally, run

.. code-block:: sh

  cd $PYSTICHE_ROOT
  tox -e docs-html

To build the latex (PDF) documentation locally, run

.. code-block:: sh

  cd $PYSTICHE_ROOT
  tox -e docs-latex

To build both, run

.. code-block:: sh

  cd $PYSTICHE_ROOT
  tox -f docs

.. note::

  Building the documentation triggers a
  `sphinx gallery <https://sphinx-gallery.github.io/stable/index.html>`_ build by
  default for the example galleries. This which will take some time to complete. To get 
  around this, ``pystiche`` offers two environment variables:

  - ``PYSTICHE_PLOT_GALLERY``: If ``False``, the code inside the galleries is not
    executed. See the
    `official sphinx-gallery documentation <https://sphinx-gallery.github.io/stable/configuration.html#without-execution>`_
    for details. Defaults to ``True``.
  - ``PYSTICHE_DOWNLOAD_GALLERY``: If ``True``, downloads pre-built
    galleries and uses them instead of rebuilding. For the ``master`` the galleries are
    at most six hours old. Defaults to ``False``.

  Both environment variables are evaluated with :func:`~distutils.util.strtobool`.
