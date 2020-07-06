.. role:: hidden
    :class: hidden-section

combustion.models
===================================


Model implementations

.. contents:: combustion.models
    :depth: 4
    :local:
    :backlinks: top
    

Convolutional Models
----------------------------------

EfficientNet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: combustion.models.EfficientNet2d
    :members: extract_features,forward

.. class:: combustion.models.EfficientNet1d

  1d variant of :class:`combustion.models.EfficientNet2d`

.. class:: combustion.models.EfficientNet3d

  3d variant of :class:`combustion.models.EfficientNet2d`

EfficientDet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: combustion.models.EfficientDet2d
    :members: extract_features,forward

.. class:: combustion.models.EfficientDet3d

  3d variant of :class:`combustion.models.EfficientDet2d`

.. class:: combustion.models.EfficientDet1d

  1d variant of :class:`combustion.models.EfficientDet2d`
