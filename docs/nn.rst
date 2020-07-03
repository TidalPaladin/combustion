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


Larger Modules
----------------------------------

.. autoclass:: combustion.nn.BiFPN
    :members: 
    :undoc-members: forward

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
