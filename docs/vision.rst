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

.. autofunction:: combustion.vision.append_bbox_label
.. autofunction:: combustion.vision.batch_box_target
.. autofunction:: combustion.vision.combine_bbox_scores_class
.. autofunction:: combustion.vision.combine_box_target
.. autofunction:: combustion.vision.filter_bbox_classes
.. autofunction:: combustion.vision.flatten_box_target
.. autofunction:: combustion.vision.split_bbox_scores_class
.. autofunction:: combustion.vision.split_box_target
.. autofunction:: combustion.vision.unbatch_box_target

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
