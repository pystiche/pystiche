``pystiche``
============

.. automodule:: pystiche

.. autofunction:: home

Objects
-------

.. autoclass:: ComplexObject
  :members:
    _properties,
    extra_properties,
    properties,
    _named_children,
    extra_named_children,
    named_children,
.. autoclass:: LossDict
  :members:
    __setitem__,
    aggregate,
    total,
    backward,
    item,
    __mul__
.. autoclass:: Module
  :members: torch_repr

Math
----

.. autofunction:: nonnegsqrt
.. autofunction:: gram_matrix
.. autofunction:: cosine_similarity
