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

Callbacks
----------------------------------

.. autoclass:: combustion.lightning.callbacks.TorchScriptCallback
    :members:

.. autoclass:: combustion.lightning.callbacks.CountMACs
    :members:

.. autoclass:: combustion.lightning.callbacks.VisualizeCallback

.. autoclass:: combustion.lightning.callbacks.KeypointVisualizeCallback

.. autoclass:: combustion.lightning.callbacks.BlendVisualizeCallback

Metrics
----------------------------------
.. autoclass:: combustion.lightning.metrics.AUROC
    :members:
.. autoclass:: combustion.lightning.metrics.BoxAUROC
    :members:
.. autoclass:: combustion.lightning.metrics.BoxAveragePrecision
    :members:
