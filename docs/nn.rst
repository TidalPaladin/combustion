.. role:: hidden
    :class: hidden-section

combustion.nn
===================================

Extensions to ``torch.nn``, ranging from fundamental layers
up to larger building blocks.

.. contents:: combustion.nn
    :depth: 4
    :local:
    :backlinks: top
    

Activation Functions
----------------------------------

.. autoclass:: combustion.nn.Swish

.. autoclass:: combustion.nn.HardSwish

.. autoclass:: combustion.nn.HardSigmoid

Convolution Layers
----------------------------------

.. autoclass:: combustion.nn.Bottleneck1d
    :members: forward
    :undoc-members: forward

.. autoclass:: combustion.nn.Bottleneck2d
    :members: forward
    :undoc-members: forward

.. autoclass:: combustion.nn.Bottleneck3d
    :members: forward
    :undoc-members: forward

.. autoclass:: combustion.nn.BottleneckFactorized2d
    :members: forward
    :undoc-members: forward

.. autoclass:: combustion.nn.BottleneckFactorized3d
    :members:
    :undoc-members: forward

Dropout Layers
----------------------------------

.. autoclass:: combustion.nn.DropConnect

Larger Modules
----------------------------------

BiFPN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: combustion.nn.BiFPN2d
    :members: 
    :undoc-members: forward

.. class:: combustion.nn.BiFPN

  Alias for :class:`combustion.nn.BiFPN2d`

  .. warning::
    This class is deprecated. Please use :class:`combustion.nn.BiFPN2d` instead

.. class:: combustion.nn.BiFPN1d

  1d variant of :class:`combustion.nn.BiFPN2d`

.. class:: combustion.nn.BiFPN3d

  3d variant of :class:`combustion.nn.BiFPN2d`

Global Attention Upsample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: combustion.nn.AttentionUpsample2d

.. class:: combustion.nn.AttentionUpsample1d

  1d version of :class:`combustion.nn.AttentionUpsample2d`.

.. class:: combustion.nn.AttentionUpsample3d

  3d version of :class:`combustion.nn.AttentionUpsample2d`.

MobileNetV3 Inverted Bottleneck
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: combustion.nn.MobileNetConvBlock2d
  :members: from_config

.. autoclass:: combustion.nn.MobileNetBlockConfig

.. class:: combustion.nn.MobileNetConvBlock1d

  1d version of :class:`combustion.nn.MobileNetConvBlock2d`.

.. class:: combustion.nn.MobileNetConvBlock3d

  3d version of :class:`combustion.nn.MobileNetConvBlock2d`.

Object Contextual Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: combustion.nn.OCR
  :members: create_region_target

Squeeze and Excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: combustion.nn.SqueezeExcite1d
.. autoclass:: combustion.nn.SqueezeExcite2d
.. autoclass:: combustion.nn.SqueezeExcite3d

Reduced Atrous Spatial Pyramid Pooling (R-ASPP Lite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: combustion.nn.RASPPLite2d

.. class:: combustion.nn.RASPPLite1d

  1d version of :class:`combustion.nn.RASPPLite2d`.

.. class:: combustion.nn.RASPPLite3d

  3d version of :class:`combustion.nn.RASPPLite2d`.

Loss Functions
----------------------------------

.. autoclass:: combustion.nn.CenterNetLoss
    :members:

Focal Loss 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: combustion.nn.focal_loss_with_logits
.. autofunction:: combustion.nn.focal_loss
.. autofunction:: combustion.nn.categorical_focal_loss

.. autoclass:: combustion.nn.FocalLoss
    :members:

.. autoclass:: combustion.nn.FocalLossWithLogits
    :members:

.. autoclass:: combustion.nn.CategoricalFocalLoss
    :members:


Utilities
----------------------------------

.. autoclass:: combustion.nn.Standardize
    :members: 

.. autoclass:: combustion.nn.DynamicSamePad
    :exclude-members: forward, extra_repr

.. autoclass:: combustion.nn.MatchShapes
    :members: forward
    :exclude-members: extra_repr
