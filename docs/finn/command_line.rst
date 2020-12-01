*******************
Command Line Entry
*******************

.. note:: **This website is currently under construction.**


Although FINN is primarily *compiler infrastructure* that provides the capabilities
researchers can use to explore custom QNN inference, we also provide
two command line entry points for productivity and ease-of-use:

* *Simple dataflow build mode:* Best-effort dataflow build by JSON build config file to convert your ONNX model.
* *Advanced build mode:* Provide your own build script with full flexibility

.. warning::
  If you are using a neural network with a topology that is substantially
  different to the FINN end-to-end examples, the simple dataflow build flow used by the compiler
  is likely to fail. For those cases, we recommend making a copy of the end-to-end
  Jupyter notebook as a starting point, visualizing the model at intermediate
  steps and adding calls to new transformations as needed.
  Once you have a working flow, you can implement a command line entry for this
  by using the "advanced mode" described here.


Simple dataflow build mode
--------------------------

This mode is intended for simpler networks whose topologies resemble the
FINN end-to-end examples.
It runs a fixed build flow spanning tidy-up, streamlining, HLS conversion
and hardware synthesis.
It can be configured to produce different outputs, including stitched IP for
integration in Vivado IPI as well as bitfiles.

To use it, first create a folder with the necessary configuration and model files:

1. Create a new folder for the dataflow build. It's best to keep this folder
   outside the FINN repo folder for cleaner separation. Let's call this folder
   ``dataflow_build_dir``.
2. Put your ONNX model to be converted under ``dataflow_build_dir/model.onnx``.
   The filename is important and must exactly be ``model.onnx``.
3. Create a JSON file with the build configuration. It must be named ``dataflow_build_dir/dataflow_build_config.json``.
   Read more about the build configuration options on :py:mod:``finn.util.build_dataflow.DataflowBuildConfig``.
   You can find an example .json file under ``src/finn/qnn-data/build_dataflow/dataflow_build_config.json``
4. Create a JSON file with the folding configuration. It must be named ``dataflow_build_dir/folding_config.json``.
   You can find an example .json file under ``src/finn/qnn-data/build_dataflow/folding_config.json``

Now you can invoke the simple dataflow build as follows:

::

  ./run_docker.sh build_dataflow <path/to/dataflow_build_dir/>

Depending on the chosen output products, the dataflow build will run for a while.

.. code-block:: none

  Building dataflow accelerator from /tmp/finn_dev_maltanar/test_build_dataflow_directory_9w2y3rh_/build_dataflow/model.onnx
  Outputs will be generated at output_tfc_w1a1_Pynq-Z1
  Running step: step_tidy_up [1/10]
  Running step: step_streamline [2/10]
  Running step: step_convert_to_hls [3/10]
  Running step: step_create_dataflow_partition [4/10]
  Running step: step_apply_folding_config [5/10]
  Running step: step_hls_ipgen [6/10]
  Running step: step_set_fifo_depths [7/10]
  Running step: step_create_stitched_ip [8/10]
  Running step: step_make_pynq_driver [9/10]
  Running step: step_synthesize_bitfile [10/10]
  Completed successfully

You will find the generated outputs under the subfolder you specified in the
build configuration, which can include the following folders and files
depending on the chosen output products:

* If the output products include :py:mod:`finn.util.build_dataflow.DataflowOutputType.BITFILE`:

  * ``bitfile/finn-accel.(bit|xclbin)`` -- generated bitfile depending on platform
  * ``bitfile/resources.xml`` -- FPGA resource utilization after synthesis
  * ``bitfile/timing.rpt`` -- Post-placement timing report

* If the output products include :py:mod:`finn.util.build_dataflow.DataflowOutputType.STITCHED_IP`:

  * ``stitched_ip/finn_vivado_stitch_proj.xpr`` -- Vivado project (including Vivado IP Integrator block design) to generate the stitched IP
  * ``stitched_ip/ip`` -- exported Vivado IP for the stitched design

* If the output products include :py:mod:`finn.util.build_dataflow.DataflowOutputType.PYNQ_DRIVER`:

  * ``driver/driver.py`` -- Python driver that can be used on PYNQ on Zynq or Alveo platforms to launch the accelerator.

Other generated files will include:

* ``time_per_step.txt`` will list the time (in seconds) each build step took
* ``build_dataflow.log`` is the build logfile that will contain any warnings/errors
* ``intermediate_models/`` will contain the ONNX file(s) produced after each build step

Advanced mode
--------------

In other cases, you may want to have more control over the build process to
implement your own FINN flow with a different combination of compilation steps,
applying preprocessing to the model, calling custom transformations and so on.
This is possible by using the `build_custom` entry as follows:

1. Create a new folder for the custom build. It's best to keep this folder
outside the FINN repo folder for cleaner separation. Let's call this folder
``custom_build_dir``.

2. Create a ``custom_build_dir/build.py`` file that will perform the build when
executed. You should also put any ONNX model(s) or other Python modules you
may want to include in your build flow in this folder (so that they get mounted
into the Docker container while building). Besides the filename and data placement,
you have complete freedom on how to implement the build flow here, including
making calls to FINN library functions, preprocessing and altering models, building several variants etc.
You can find a basic example of build.py under ``src/finn/qnn-data/build_dataflow/build.py``.

You can launch the custom build flow using:

::

 ./run_docker.sh build_custom <path/to/custom_build_dir/>

This will mount the specified folder into the FINN Docker container and launch
your ``build.py``.
