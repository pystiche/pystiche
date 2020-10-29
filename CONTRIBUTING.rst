Contributing
============

First and foremost: Thank you for your interest in ``pystiche`` s development! We
appreciate all contributions be it code or something else.

Guide lines
-----------

``pystiche`` uses the `GitHub workflow <https://guides.github.com/introduction/flow/>`_
. Below is small guide how to make your first contribution.

.. note::

  The following guide assumes that `git <https://git-scm.com/>`_,
  `python <https://www.python.org/>`_, and `pip <https://pypi.org/project/pip/>`_ ,
  are available on your system. If that is not the case, follow the official
  installation instructions.

.. note::

  ``pystiche`` officially supports Python ``3.6``, ``3.7``, and ``3.8``. To ensure
  backwards compatibility, the development should happen on the minimum Python
  version, i. e. ``3.6``.

1. Fork ``pystiche`` on GitHub

  Navigate to `pmeier/pystiche <https://github.com/pmeier/pystiche>`_ on GitHub and
  click the **Fork** button in the top right corner.

2. Clone your fork to your local file system

  Use ``git clone`` to get a local copy of ``pystiche`` s repository that you can work
  on:

  .. code-block:: sh

    $ PYSTICHE_ROOT="pystiche"
    $ git clone "https://github.com/pmeier/pystiche.git" $PYSTICHE_ROOT

3. Setup your development environment

  .. code-block:: sh

    $ cd $PYSTICHE_ROOT
    $ virtualenv .venv --prompt="(pystiche) "
    $ source .venv/bin/activate
    $ pip install -r requirements-dev.txt
    $ pre-commit install

  .. note::

    While ``pystiche`` s development requirements are fairly lightweight, it is still
    recommended to install them in a virtual environment rather than system wide. If you
    do not have ``virtualenv`` installed, you can do so by running
    ``pip install --user virtualenv``.

4. Create a branch for local development

  Use ``git checkout`` to create local branch with a descriptive name:

  .. code-block:: sh

    $ PYSTICHE_BRANCH="my-awesome-feature-or-bug-fix"
    $ git checkout -b $PYSTICHE_BRANCH

  Now make your changes. Happy Coding!

5. Use ``tox`` to run various checks

  .. code-block:: sh

    $ tox

  .. note::

    Running ``tox`` is equivalent to running

    .. code-block:: sh

      $ tox -e lint-style
      $ tox -e lint-typing
      $ tox -e tests-integration
      $ tox -e tests-galleries
      $ tox -e tests-docs

    You can find details what the individual commands do below of this guide.

6. Commit and push your changes

  If all checks are passing you can commit your changes an push them to your fork:

  .. code-block:: sh

    $ git add .
    $ git commit -m "Descriptive message of the changes made"
    $ git push -u origin $PYSTICHE_BRANCH

  .. note::

    For larger changes, it is good practice to split them in multiple small commits
    rather than one large one. If you do that, make sure to run the test suite before
    every commit. Furthermore, use ``git push`` without any parameters for consecutive
    commits.

7. Open a Pull request (PR)

  1. Navigate to `pmeier/pystiche/pulls <https://github.com/pmeier/pystiche/pulls>`_ on
     GitHub and click on the green button "New pull request".
  2. Click on "compare across forks" below the "Compare changes" headline.
  3. Select your fork for "head repository" and your branch for "compare" in the
     drop-down menus.
  4. Click the the green button "Create pull request".

  .. note::

    If the time between the branch being pushed and the PR being opened is not too
    long, GitHub will offer you a yellow box after step 1. If you click the button,
    you can skip steps 2. and 3.

.. note::

  Steps 1. to 3. only have to performed once. If you want to continue contributing,
  make sure to branch from the current ``master`` branch. You can use ``git pull``

  .. code-block:: sh

    $ git checkout master
    $ git pull origin
    $ git checkout -b "my-second-awesome-feature-or-bug-fix"

  If you forgot to do that or if since the creation of your branch many commits have
  been made to the ``master`` branch, simply rebase your branch on top of it.

  .. code-block:: sh

    $ git checkout master
    $ git pull origin
    $ git checkout "my-second-awesome-feature-or-bug-fix"
    $ git rebase master

If you are contributing bug-fixes or
documentation improvements, you can open a
`pull request (PR) <https://github.com/pmeier/pystiche/pulls>`_ without further
discussion. If on the other hand you are planning to contribute new features, please
open an `issue <https://github.com/pmeier/pystiche/issues>`_ and discuss the feature
with us first.

Every PR is subjected to multiple automatic checks (continuous integration, CI) as well
as a manual code review that it has to pass before it can be merged. The automatic
checks are performed by `tox <https://tox.readthedocs.io/en/latest/>`_. You can find
details and instructions how to run the checks locally below.

Code format and linting
-----------------------

``pystiche`` uses `isort <https://timothycrosley.github.io/isort/>`_ to sort the
imports, `black <https://black.readthedocs.io/en/stable/>`_ to format the code, and
`flake8 <https://flake8.pycqa.org/en/latest/>`_ to enforce
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ compliance. To format and check the
code style, run

.. code-block:: sh

  cd $PYSTICHE_ROOT
  source .venv/bin/activate
  tox -e lint-style

.. note::

  Amongst others, ``isort``, ``black``, and ``flake8`` are run by
  `pre-commit <https://pre-commit.com/>`_ before every commit.

Furthermore, ``pystiche_papers`` is
`PEP561 <https://www.python.org/dev/peps/pep-0561/>`_ compliant and checks the type
annotations with `mypy <http://mypy-lang.org/>`_. To check the static typing, run

.. code-block:: sh

  cd $PYSTICHE_ROOT
  source .venv/bin/activate
  tox -e lint-typing

For convenience, you can run all lint checks with

.. code-block:: sh

  cd $PYSTICHE_ROOT
  source .venv/bin/activate
  tox -f lint


Test suite
----------

``pystiche`` uses `pytest <https://docs.pytest.org/en/stable/>`_ to run the test suite.
You can run it locally with

.. code-block:: sh

  cd $PYSTICHE_ROOT
  source .venv/bin/activate
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
    source .venv/bin/activate
    tox -- --skip-large-download


Documentation
-------------

To build the html documentation locally, run

.. code-block:: sh

  cd $PYSTICHE_ROOT
  source .venv/bin/activate
  tox -e docs-html

To build the latex (PDF) documentation locally, run

.. code-block:: sh

  cd $PYSTICHE_ROOT
  source .venv/bin/activate
  tox -e docs-latex

To build both, run

.. code-block:: sh

  cd $PYSTICHE_ROOT
  source .venv/bin/activate
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
