.. role:: hidden
    :class: hidden-section

combustion.util
===================================

Misc utils

Mask Ops
----------------------------------
.. autofunction:: combustion.util.masks.connect_masks
.. autofunction:: combustion.util.masks.contract_mask
.. autofunction:: combustion.util.masks.edge_dist
.. autofunction:: combustion.util.masks.expand_mask
.. autofunction:: combustion.util.masks.get_edges
.. autofunction:: combustion.util.masks.get_instances
.. autofunction:: combustion.util.masks.get_adjacency
.. autofunction:: combustion.util.masks.index_assign_mask
.. autofunction:: combustion.util.masks.index_mask
.. autofunction:: combustion.util.masks.inverse_edge_dist
.. autofunction:: combustion.util.masks.mask_to_box
.. autofunction:: combustion.util.masks.mask_to_polygon
.. autofunction:: combustion.util.masks.min_spacing


Others
----------------------------------
.. autofunction:: combustion.util.ntuple
.. autofunction:: combustion.util.percent_change
.. autofunction:: combustion.util.percent_error_change

Validation
----------------------------------
Validation methods are not yet TorchScript compatible.

.. autofunction:: combustion.util.check_dimension
.. autofunction:: combustion.util.check_dimension_match
.. autofunction:: combustion.util.check_dimension_within_range
.. autofunction:: combustion.util.check_is_array
.. autofunction:: combustion.util.check_is_tensor
.. autofunction:: combustion.util.check_ndim
.. autofunction:: combustion.util.check_ndim_match
.. autofunction:: combustion.util.check_ndim_within_range
.. autofunction:: combustion.util.check_shape
.. autofunction:: combustion.util.check_shapes_match

Visualization
----------------------------------

.. autofunction:: combustion.util.apply_colormap
.. autofunction:: combustion.util.alpha_blend
