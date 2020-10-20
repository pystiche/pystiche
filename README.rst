.. start-badges

.. list-table::
    :stub-columns: 1

    * - package
      - |license| |status|
    * - citation
      - |pyopensci| |joss|
    * - code
      - |black| |mypy| |lint|
    * - tests
      - |tests| |coverage|
    * - docs
      - |docs| |rtd|

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

.. image:: docs/source/graphics/banner/banner.jpg
    :alt: pystiche banner

``pystiche`` has similar goals as Deep Learning (DL) frameworks such as PyTorch:

1. **Accessibility**
    Starting off with NST can be quite overwhelming due to the sheer amount of
    techniques one has to know and be able to deploy. ``pystiche`` aims to provide an
    easy-to-use interface that reduces the necessary prior knowledge about NST and DL
    to a minimum.
2. **Reproducibility**
    Implementing NST from scratch is not only inconvenient but also error-prone.
    ``pystiche`` aims to provide reusable tools that let developers focus on their
    ideas rather than worrying about bugs in everything around it.


Installation
============

``pystiche`` is a proper Python package and can be installed with ``pip``. The latest
release can be installed with

.. code-block:: sh

  pip install pystiche

Usage
=====

``pystiche`` makes it easy to define the optimization criterion for an NST task fully
compatible with PyTorch. For example, the banner above was generated with the following
``criterion``:

.. code-block:: python

  from pystiche import enc, loss, ops

  multi_layer_encoder = enc.vgg19_multi_layer_encoder()

  criterion = loss.PerceptualLoss(
      content_loss=ops.FeatureReconstructionOperator(
          multi_layer_encoder.extract_encoder("relu4_2")
      ),
      style_loss=ops.MultiLayerEncodingOperator(
          multi_layer_encoder,
          ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"),
          lambda encoder, layer_weight: ops.GramOperator(
              encoder, score_weight=layer_weight
          ),
          score_weight=1e3,
      ),
  )

For the full example, head over to the example
`NST with pystiche <https://pystiche.readthedocs.io/en/latest/galleries/examples/beginner/example_nst_with_pystiche.html>`_.

Documentation
=============

For

- `detailed installation instructions <https://pystiche.readthedocs.io/en/latest/getting_started/installation.html>`_,
- a `gallery of usage examples <https://pystiche.readthedocs.io/en/latest/galleries/examples/index.html>`_,
- the `API reference <https://pystiche.readthedocs.io/en/latest/api/index.html>`_,
- the `contributing guidelines <https://pystiche.readthedocs.io/en/latest/getting_started/contributing.html>`_,

or anything else, head over to the `documentation <https://pystiche.readthedocs.io/en/latest/>`_.

Citation
========

If you use this software, please cite it as

.. code-block:: bibtex

  @Article{ML2020,
    author  = {Meier, Philip and Lohweg, Volker},
    journal = {Journal of Open Source Software {JOSS}},
    title   = {pystiche: A Framework for Neural Style Transfer},
    year    = {2020},
    doi     = {10.21105/joss.02761},
  }

.. |license|
  image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: License

.. |status|
  image:: https://www.repostatus.org/badges/latest/active.svg
    :target: https://www.repostatus.org/#active
    :alt: Project Status: Active

.. |pyopensci|
  image:: https://tinyurl.com/y22nb8up
    :target: https://github.com/pyOpenSci/software-review/issues/25
    :alt: pyOpenSci

.. |joss|
  image:: https://joss.theoj.org/papers/10.21105/joss.02761/status.svg
    :target: https://doi.org/10.21105/joss.02761
    :alt: JOSS

.. |black|
  image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black

.. |mypy|
  image:: http://www.mypy-lang.org/static/mypy_badge.svg
    :target: http://mypy-lang.org/
    :alt: mypy

.. |lint|
  image:: https://github.com/pmeier/pystiche/workflows/lint/badge.svg
    :target: https://github.com/pmeier/pystiche/actions?query=workflow%3Alint+branch%3Amaster
    :alt: Lint status via GitHub Actions

.. |tests|
  image:: https://github.com/pmeier/pystiche/workflows/tests/badge.svg
    :target: https://github.com/pmeier/pystiche/actions?query=workflow%3Atests+branch%3Amaster
    :alt: Test status via GitHub Actions

.. |coverage|
  image:: https://codecov.io/gh/pmeier/pystiche/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pmeier/pystiche
    :alt: Test coverage

.. |docs|
  image:: https://github.com/pmeier/pystiche/workflows/docs/badge.svg
    :target: https://github.com/pmeier/pystiche/actions?query=workflow%3Adocs+branch%3Amaster
    :alt: Docs status via GitHub Actions

.. |rtd|
  image:: https://img.shields.io/readthedocs/pystiche?label=latest&logo=read%20the%20docs
    :target: https://pystiche.readthedocs.io/en/latest/?badge=latest
    :alt: Latest documentation hosted on Read the Docs
