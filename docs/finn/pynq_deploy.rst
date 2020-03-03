.. _pynq_deploy:

***************
PYNQ Deployment
***************

.. note:: **This website is currently under construction.**

.. image:: /img/pynq-deploy.png
   :scale: 70%
   :align: center

This chapter is about the hardware generation and deployment on PYNQ. If you need more information about PYNQ, please have a look at the `PYNQ website <https://pynq.readthedocs.io/en/v2.5.1/>`_.

Create PYNQ Shell Project
=========================

To deploy the network on A PYNQ platform, it needs to be put inside an appropriate *shell*. This *shell* bridges the network with the interfaces the underlying system exposes. This can be done using the transformation MakePYNQProject, see :py:mod:`finn.transformation.fpgadataflow.make_pynq_proj.MakePYNQProject`. 

Test on Hardware
================

Synthesis, Place and Route
--------------------------

After integrating the model into the PYNQ shell, Vivado *Synthesis, Place and Route* can be launched. The result is a bitfile which can be used for the PYNQ board. In FINN this can be done using a transformation pass. For details, please have a look at :py:mod:`finn.transformation.fpgadataflow.synth_pynq_proj.SynthPYNQProject`.

Generate PYNQ runtime code
--------------------------

Additionally, a Python code is necessary to execute the model on the board. This is done by transformation pass :py:mod:`finn.transformation.fpgadataflow.make_pynq_driver.MakePYNQDriver`. 

Deployment and Remote Execution
-------------------------------

The bitfile and the driver file(s) are copied to the PYNQ board and can be executed there using the *onnx_exec* function with the right *exec_mode* settings. For details please have a look at transformation :py:mod:`finn.transformation.fpgadataflow.make_deployment.DeployToPYNQ` and the execution function :py:mod:`finn.core.onnx_exec`.
