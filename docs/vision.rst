.. role:: hidden
    :class: hidden-section

combustion.vision
===================================


.. contents:: combustion.vision
    :depth: 2
    :local:
    :backlinks: top
    

.. currentmodule:: combustion.vision

Conversions
----------------------------------

.. autofunction:: combustion.vision.to_8bit

Filters
----------------------------------

.. autoclass:: combustion.vision.filters.CLAHE
    :members: 

.. autofunction:: combustion.vision.filters.relative_intensity
.. autoclass:: combustion.vision.filters.RelativeIntensity
    :members: 
    :exclude-members: extra_repr

.. autofunction:: combustion.vision.filters.gaussian_blur2d
.. autoclass:: combustion.vision.filters.GaussianBlur2d
    :members: 
    :exclude-members: extra_repr


Ops
----------------------------------

.. autofunction:: combustion.vision.nms
.. autofunction:: combustion.vision.visualize_bbox

.. autoclass:: combustion.vision.AnchorsToPoints
    :members: 

.. autoclass:: combustion.vision.PointsToAnchors
    :members: 

.. autoclass:: combustion.vision.ConfusionMatrixIoU
    :members: 

.. autoclass:: combustion.vision.BinaryLabelIoU
    :members: 


CenterNet
----------------------------------
.. autoclass:: combustion.vision.centernet.CenterNetMixin
    :members: 
