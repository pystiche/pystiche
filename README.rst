.. start-badges

.. list-table::
    :stub-columns: 1

    * - package
      - |license| |status|
    * - code
      - |black| |mypy| |lint|
    * - tests
      - |linux_macos| |windows| |coverage|
    * - docs
      - |docs|

.. end-badges


``pystiche``
============

``pystiche`` (pronounced
`/ˈpaɪˈstiʃ/ <http://ipa-reader.xyz/?text=%CB%88pa%C9%AA%CB%88sti%CA%83>`_ ) is a
framework for
`Neural Style Transfer (NST) <https://github.com/ycjing/Neural-Style-Transfer-Papers>`_
built upon `PyTorch <https://pytorch.org>`_. The name of the project is a pun on
*pastiche* `meaning <https://en.wikipedia.org/wiki/Pastiche>`_:

    A pastiche is a work of visual art [...] that imitates the style or character of
    the work of one or more other artists. Unlike parody, pastiche celebrates, rather
    than mocks, the work it imitates.


Getting started
---------------

``pystiche`` is a proper Python package and can be installed with ``pip``. It is not
yet listed on `PyPI <https://pypi.org/>`_, since it will be reviewed at
`pyOpenSci <https://github.com/pmeier/pystiche/issues/93>`_ . To install the latest
version of ``pystiche`` run

.. code-block:: sh

  pip install git+https://github.com/pmeier/pystiche


For extended installation instructions and usage examples please consult the
documentation `hosted on readthedocs.com <https://pystiche.readthedocs.io/en/latest>`_ .


.. |license|
  image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: License

.. |status|
  image:: https://www.repostatus.org/badges/latest/active.svg
    :target: https://www.repostatus.org/#active
    :alt: Project Status: Active

.. |black|
  image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black

.. |mypy|
  image:: http://www.mypy-lang.org/static/mypy_badge.svg
    :target: http://mypy-lang.org/
    :alt: mypy

.. |lint|
  image:: https://github.com/pmeier/pystiche/workflows/Lint/badge.svg
    :target: https://github.com/pmeier/pystiche/actions?query=workflow%3ALint+branch%3Amaster
    :alt: Lint status via GitHub Actions

.. |linux_macos|
  image:: https://img.shields.io/travis/com/pmeier/pystiche?label=Linux%20%2F%20macOS&logo=Travis
    :target: https://travis-ci.com/pmeier/pystiche
    :alt: Test status on Linux and macOS via Travis CI

.. |windows|
  image:: https://img.shields.io/appveyor/build/pmeier/pystiche?label=Windows&logo=AppVeyor
    :target: https://ci.appveyor.com/project/pmeier/pystiche
    :alt: Test status on Windows via AppVeyor

.. |coverage|
  image:: https://codecov.io/gh/pmeier/pystiche/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pmeier/pystiche
    :alt: Test coverage

.. |docs|
  image:: https://readthedocs.org/projects/pystiche/badge/?version=latest
    :target: https://pystiche.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation status
