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

.. autoclass:: combustion.nn.BiFPN
    :members: 
    :undoc-members: forward


.. autoclass:: combustion.nn.MobileNetConvBlock2d

.. class:: combustion.nn.MobileNetConvBlock1d

  1d version of :class:`combustion.nn.MobileNetConvBlock2d`.

.. class:: combustion.nn.MobileNetConvBlock3d

  3d version of :class:`combustion.nn.MobileNetConvBlock2d`.

.. autoclass:: combustion.nn.SqueezeExcite1d
.. autoclass:: combustion.nn.SqueezeExcite2d
.. autoclass:: combustion.nn.SqueezeExcite3d



Loss Functions
----------------------------------

.. autofunction:: combustion.nn.focal_loss_with_logits
.. autofunction:: combustion.nn.focal_loss

.. autoclass:: combustion.nn.FocalLoss
    :members:

.. autoclass:: combustion.nn.FocalLossWithLogits
    :members:

.. autoclass:: combustion.nn.CenterNetLoss
    :members:


Utilities
----------------------------------

.. autoclass:: combustion.nn.Standardize
    :members: 
    :undoc-members: forward
