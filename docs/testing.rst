.. role:: hidden
    :class: hidden-section

combustion.testing
===================================

Extensions to ``torch.nn``, ranging from fundamental layers
up to larger building blocks.

.. contents:: combustion.testing
    :depth: 2
    :local:
    :backlinks: top
    

.. currentmodule:: combustion.testing


Assertions
----------------------------------

.. autofunction:: combustion.testing.assert_has_gradient
.. autofunction:: combustion.testing.assert_in_eval_mode
.. autofunction:: combustion.testing.assert_in_training_mode
.. autofunction:: combustion.testing.assert_is_int_tensor
.. autofunction:: combustion.testing.assert_tensors_close


PyTest Decorators
----------------------------------

.. autofunction:: combustion.testing.cuda_or_skip

LightningModuleTest
----------------------------------

In order to facilitate model testing without excessive boilerplate code,
a pytest base class is provided that attempts to provide a set of minimal
tests for a PyTorch Lightning ``LightningModule``. By implementing a
small number of fixtures that provide a model to be tested and data
that can be used for testing, many stages of the ``LightningModule`` 
lifecycle can be tested without writing any additional test code.

.. autoclass:: combustion.testing.LightningModuleTest
    :members:


Mixins
----------------------------------

.. autoclass:: combustion.testing.TorchScriptTestMixin
    :members:

.. autoclass:: combustion.testing.TorchScriptTraceTestMixin
    :members:
