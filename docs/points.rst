.. role:: hidden
    :class: hidden-section

combustion.points
===================================

Operations for point clouds.

.. contents:: combustion.points
    :depth: 4
    :local:
    :backlinks: top
    

Transforms
----------------------------------

.. autoclass:: combustion.points.Rotate

.. function:: combustion.points.rotate

    Rotates a collection of points using rotation values in radians or degrees.
    See :class:`combustion.points.Rotate` for more details.

.. autoclass:: combustion.points.CenterCrop

.. function:: combustion.points.center_crop

    Crops a point cloud to a given size about the origin.
    See :class:`combustion.points.CenterCrop` for more details.

.. autofunction:: combustion.points.projection_mask
    

Randomized Transforms
----------------------------------
.. autoclass:: combustion.points.RandomRotate
.. function:: combustion.points.random_rotate

    Rotates a collection of points randomly between a minimum and maximum possible rotation.
    See :class:`combustion.points.RandomRotate` for more details.
