``pystiche.ops``
================

.. automodule:: pystiche.ops

.. autoclass:: Operator
  :members: set_input_guide, has_input_guide, apply_guide
  :undoc-members:

.. autoclass:: RegularizationOperator
  :members: process_input_image
  :undoc-members:
.. autoclass:: ComparisonOperator
  :members:
    set_target_guide,
    has_target_guide,
    set_target_image,
    has_target_image,
    process_input_image
  :undoc-members:
.. autoclass:: PixelOperator
  :members: process_input_image
  :undoc-members:
.. autoclass:: EncodingOperator
  :members: process_input_image
  :undoc-members:

.. autoclass:: PixelRegularizationOperator
  :members:
    set_input_guide,
    has_input_guide,
    input_image_to_repr,
    calculate_score
  :undoc-members:
.. autoclass:: EncodingRegularizationOperator
  :members:
    set_input_guide,
    has_input_guide,
    input_enc_to_repr,
    calculate_score
  :undoc-members:
.. autoclass:: PixelComparisonOperator
  :members:
    set_target_guide,
    has_target_guide,
    set_target_image,
    has_target_image,
    target_image_to_repr,
    set_input_guide,
    has_input_guide,
    input_image_to_repr,
    calculate_score
  :undoc-members:
.. autoclass:: EncodingComparisonOperator
  :members:
    set_target_guide,
    has_target_guide,
    set_target_image,
    has_target_image,
    target_enc_to_repr,
    set_input_guide,
    has_input_guide,
    input_enc_to_repr,
    calculate_score
  :undoc-members:


Container
---------

.. autoclass:: OperatorContainer
  :members:
    set_target_guide,
    set_target_image,
    set_input_guide
.. autoclass:: MultiLayerEncodingOperator
.. autoclass:: MultiRegionOperator
  :members:
    set_regional_target_guide,
    set_regional_target_image,
    set_regional_input_guide


Regularization
--------------

.. autoclass:: TotalVariationOperator
  :show-inheritance:


Comparison
----------

.. autoclass:: FeatureReconstructionOperator
  :show-inheritance:
.. autoclass:: GramOperator
  :show-inheritance:
.. autoclass:: MRFOperator
  :members: scale_and_rotate_transforms
  :show-inheritance:
