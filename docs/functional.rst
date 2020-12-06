.. role:: hidden
    :class: hidden-section

combustion.nn.functional
===================================

Extensions to ``torch.nn.functional``.

.. contents:: combustion.nn.functional
    :depth: 4
    :local:
    :backlinks: top
    

Activation Functions
----------------------------------

.. autofunction:: combustion.nn.functional.swish
.. autofunction:: combustion.nn.functional.hard_swish
.. autofunction:: combustion.nn.functional.hard_sigmoid

Convolution
----------------------------------
.. autofunction:: combustion.nn.functional.fourier_conv2d

Loss Functions
----------------------------------

.. class:: combustion.nn.functional.complete_iou_loss

  Functional form of :class:`combustion.nn.CompleteIoULoss`


Utilities
----------------------------------
.. autofunction:: combustion.nn.functional.patch_dynamic_same_pad
.. autofunction:: combustion.nn.functional.fill_normal
