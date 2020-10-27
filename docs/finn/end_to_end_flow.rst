***************
End-to-End Flow
***************

.. note:: **This website is currently under construction.**

The following image shows an example end-to-end flow in FINN, starting from a trained PyTorch/Brevitas network and going all the way to a running FPGA accelerator.
As you can see in the picture, FINN has a high modularity and has the property that the flow can be stopped at any point and the intermediate result can be used for further processing or other purposes. This enables a wide range of users to benefit from FINN, even if they do not use the whole flow.

.. image:: ../../notebooks/end2end_example/finn-design-flow-example.svg
   :scale: 50%
   :align: center

The white fields show the state of the network representation in the respective step. The colored fields represent the transformations that are applied to the network to achieve a certain result. The diagram is divided into five sections, each of it includes several flow steps. The flow starts in top left corner with Brevitas export (green section), followed by the preparation of the network (blue section) for the Vivado HLS and Vivado IPI (orange section). There is also a section for testing and verification in software (red section) and the hardware generation and deployment on the PYNQ board (yellow section).

This example flow is covered in the `end2end_example <https://github.com/Xilinx/finn/tree/master/notebooks/end2end_example>`_ Jupyter notebooks.
For a more detailed overview about the different flow sections, please have a look at the corresponding pages:

.. toctree::

   brevitas_export
   nw_prep
   hw_build
   verification
