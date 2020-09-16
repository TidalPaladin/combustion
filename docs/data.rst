.. role:: hidden
    :class: hidden-section

combustion.data
===================================

PLACEHOLDER

.. contents:: combustion.data
    :depth: 4
    :local:
    :backlinks: top

Base Datasets
----------------------------------

.. autoclass:: combustion.data.TransformableDataset
    :members: apply_transforms

Saving and Loading
----------------------------------

.. autofunction:: combustion.data.save_hdf5
.. autofunction:: combustion.data.save_torch

.. autoclass:: combustion.data.SerializeMixin
    :members:

.. autoclass:: combustion.data.HDF5Dataset
    :members:

.. autoclass:: combustion.data.TorchDataset
    :members:

Window Operations
----------------------------------

.. autoclass:: combustion.data.Window
    :members:

.. autoclass:: combustion.data.DenseWindow
    :members:

.. autoclass:: combustion.data.SparseWindow
    :members:

