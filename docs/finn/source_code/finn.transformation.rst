**************
Transformation
**************

If you want to know more details about transformation passes, please take a look at section ":ref:`transformation_pass`" in chapter *Internals*.

Submodules
==========

.. toctree::
   :maxdepth: 2

   finn.transformation.fpgadataflow
   finn.transformation.qonnx
   finn.transformation.streamline

Transformation Passes
======================

Base Class
----------

.. automodule:: qonnx.transformation.base
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.batchnorm\_to\_affine
------------------------------------------------

.. automodule:: qonnx.transformation.batchnorm_to_affine
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.bipolar\_to\_xnor
--------------------------------------------

.. automodule:: qonnx.transformation.bipolar_to_xnor
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.change\_3d\_tensors\_to\_4d
-------------------------------------------------

.. automodule:: qonnx.transformation.change_3d_tensors_to_4d
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.change\_batchsize
----------------------------------------

.. automodule:: qonnx.transformation.change_batchsize
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.change\_datalayout
--------------------------------------------

.. automodule:: qonnx.transformation.change_datalayout
  :members:
  :undoc-members:
  :show-inheritance:


qonnx.transformation.channels\_last
--------------------------------------------

.. automodule:: qonnx.transformation.channels_last
  :members:
  :undoc-members:
  :show-inheritance:


qonnx.transformation.create\_generic\_partitions
-------------------------------------------------

.. automodule:: qonnx.transformation.create_generic_partitions
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.double\_to\_single\_float
----------------------------------------------------

.. automodule:: qonnx.transformation.double_to_single_float
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.expose\_intermediate
------------------------------------------

.. automodule:: qonnx.transformation.expose_intermediate
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.extend\_partition
------------------------------------------

.. automodule:: qonnx.transformation.extend_partition
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.extract\_conv\_bias
------------------------------------------

.. automodule:: qonnx.transformation.extract_conv_bias
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.extract\_quant\_scale\_zeropt
----------------------------------------------------

.. automodule:: qonnx.transformation.extract_quant_scale_zeropt
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.fold\_constants
--------------------------------------

.. automodule:: qonnx.transformation.fold_constants
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.gemm\_to\_matmul
------------------------------------------

.. automodule:: qonnx.transformation.gemm_to_matmul
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.general
------------------------------

.. automodule:: qonnx.transformation.general
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.infer\_data\_layouts
-------------------------------------------

.. automodule:: qonnx.transformation.infer_data_layouts
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.infer\_datatypes
-------------------------------------------

.. automodule:: qonnx.transformation.infer_datatypes
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.infer\_shapes
----------------------------------------

.. automodule:: qonnx.transformation.infer_shapes
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.insert\_topk
---------------------------------------

.. automodule:: qonnx.transformation.insert_topk
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.lower\_convs\_to\_matmul
---------------------------------------------------

.. automodule:: qonnx.transformation.lower_convs_to_matmul
   :members:
   :undoc-members:
   :show-inheritance:

qonnx.transformation.make\_input\_chanlast
---------------------------------------------

.. automodule:: qonnx.transformation.make_input_chanlast
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.merge\_onnx\_models
----------------------------------------

.. automodule:: qonnx.transformation.merge_onnx_models
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.pruning
------------------------------

.. automodule:: qonnx.transformation.pruning
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.qcdq\_to\_qonnx
----------------------------------------

.. automodule:: qonnx.transformation.qcdq_to_qonnx
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.qonnx\_to\_qcdq
-------------------------------------

.. automodule:: qonnx.transformation.qonnx_to_qcdq
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.quant\_constant\_folding
----------------------------------------------

.. automodule:: qonnx.transformation.quant_constant_folding
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.quantize\_graph
-------------------------------------

.. automodule:: qonnx.transformation.quantize_graph
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.rebalance\_conv
----------------------------------------

.. automodule:: qonnx.transformation.rebalance_conv
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.remove
----------------------------

.. automodule:: qonnx.transformation.remove
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.resize\_conv\_to\_deconv
-----------------------------------------------

.. automodule:: qonnx.transformation.resize_conv_to_deconv
  :members:
  :undoc-members:
  :show-inheritance:

qonnx.transformation.subpixel\_to\_deconv
-----------------------------------------------

.. automodule:: qonnx.transformation.subpixel_to_deconv
  :members:
  :undoc-members:
  :show-inheritance:

finn.transformation.move\_reshape
----------------------------------------

.. automodule:: finn.transformation.move_reshape
   :members:
   :undoc-members:
   :show-inheritance:
