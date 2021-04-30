.. role:: hidden
    :class: hidden-section

combustion.lightning
===================================

Utilities to facilitate operation with 
`PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/stable/>`_

.. contents:: combustion.lightning
    :depth: 2
    :local:
    :backlinks: top


.. autoclass:: combustion.lightning.HydraMixin
    :members:

General Callbacks
----------------------------------

.. autoclass:: combustion.lightning.callbacks.SaveTensors
    :members:

.. autoclass:: combustion.lightning.callbacks.TorchScriptCallback
    :members:

.. autoclass:: combustion.lightning.callbacks.CountMACs
    :members:


Visualization Callbacks
----------------------------------
Combustion provides a set of callbacks that integrate with PyTorch Lightning
to provide a simplistic yet powerful visualization feature set. Tensors to be
visualized are attached as model attributes during training or inference steps,
and callbacks read these attributes to create visualizations.

.. autoclass:: combustion.lightning.callbacks.VisualizeCallback

.. autoclass:: combustion.lightning.callbacks.KeypointVisualizeCallback

.. autoclass:: combustion.lightning.callbacks.BlendVisualizeCallback

.. autoclass:: combustion.lightning.callbacks.ImageSave

Metrics
----------------------------------
.. autoclass:: combustion.lightning.metrics.BoxAUROC
    :members:
.. autoclass:: combustion.lightning.metrics.BoxAveragePrecision
    :members:
