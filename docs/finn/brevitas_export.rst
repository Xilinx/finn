.. _brevitas_export:

***************
Brevitas Export
***************

.. image:: /img/brevitas-export.png
   :scale: 70%
   :align: center

FINN expects an ONNX model as input. This can be a model trained with `Brevitas <https://github.com/Xilinx/brevitas>`_. Brevitas is a PyTorch library for quantization-aware training and the FINN Docker image comes with several `example Brevitas networks <https://github.com/maltanar/brevitas_cnv_lfc>`_. Brevitas provides an export of a quantized network in ONNX representation. The resulting model consists only of `ONNX standard nodes <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_, but also contains additional attributes for the ONNX nodes to represent low precision datatypes. To work with the model it is wrapped into :ref:`modelwrapper` provided by FINN. 

The model can now be further processed in FINN, the next flow step is :ref:`nw_prep`.
