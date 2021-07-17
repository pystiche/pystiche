``pystiche.loss``
=================

.. automodule:: pystiche.loss

.. autoclass:: Loss
    :members:
        set_input_guide,
        forward,
    :undoc-members:
    :show-inheritance:

.. autoclass:: RegularizationLoss
    :members:
        input_enc_to_repr,
        calculate_score,
    :undoc-members:
    :show-inheritance:

.. autoclass:: ComparisonLoss
    :members:
        set_target_image,
        input_enc_to_repr,
        target_enc_to_repr,
        calculate_score,
    :undoc-members:
    :show-inheritance:


Container
---------

.. autoclass:: LossContainer
    :members:
        set_input_guide,
        set_target_image,

.. autoclass:: SameTypeLossContainer

.. autoclass:: MultiLayerEncodingLoss

.. autoclass:: MultiRegionLoss
    :members:
        set_regional_input_guide,
        set_regional_target_image,

.. autoclass:: PerceptualLoss
    :members:
        regional_content_guide,
        set_content_guide,
        regional_style_image,
        set_style_image,


Regularization
--------------

.. autoclass:: TotalVariationLoss
    :show-inheritance:


Comparison
----------

.. autoclass:: FeatureReconstructionLoss
    :show-inheritance:
.. autoclass:: GramLoss
    :show-inheritance:
.. autoclass:: MRFLoss
    :members: scale_and_rotate_transforms
    :show-inheritance:
