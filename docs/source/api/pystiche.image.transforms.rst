``pystiche.image.transforms``
=============================

.. automodule:: pystiche.image.transforms

.. autoclass:: Transform
.. autoclass:: ComposedTransform


Color
-----

.. autoclass:: RGBToGrayscale
.. autoclass:: GrayscaleToFakegrayscale
.. autoclass:: RGBToFakegrayscale
.. autoclass:: GrayscaleToBinary
.. autoclass:: RGBToBinary
.. autoclass:: RGBToYUV
.. autoclass:: YUVToRGB


Crop
----

.. autoclass:: TopLeftCrop
.. autoclass:: BottomLeftCrop
.. autoclass:: TopRightCrop
.. autoclass:: BottomRightCrop
.. autoclass:: CenterCrop
.. autoclass:: ValidRandomCrop


I/O
---

.. autoclass:: ImportFromPIL
.. autoclass:: ExportToPIL


Miscellaneous
-------------

.. autoclass:: FloatToUint8Range
.. autoclass:: Uint8ToFloatRange
.. autoclass:: ReverseChannelOrder
.. autoclass:: Normalize
.. autoclass:: Denormalize


Motif
-----

.. autoclass:: TransformMotifAffinely
.. autoclass:: ShearMotif
.. autoclass:: RotateMotif
.. autoclass:: ScaleMotif
.. autoclass:: TranslateMotif


Processing
----------

.. autoclass:: TorchPreprocessing
.. autoclass:: TorchPostprocessing
.. autoclass:: CaffePreprocessing
.. autoclass:: CaffePostprocessing


Resize
------

.. autoclass:: Resize
.. autoclass:: Rescale


Functional
----------

.. automodule:: pystiche.image.transforms.functional

.. autofunction:: resize
