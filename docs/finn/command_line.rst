.. _command_line:

*******************
Command Line Entry
*******************

Although FINN is primarily *compiler infrastructure* that provides the capabilities
researchers can use to explore custom QNN inference, we also provide
two command line entry points for productivity and ease-of-use:

* *Simple dataflow build mode:* Best-effort dataflow build by JSON build config file to convert your ONNX model.
* *Advanced build mode:* Provide your own build script with full flexibility

.. note:: **When setting up builds using either build mode, you should keep all required data (model, config files etc.) inside the build folder and not use symlinks.**

.. warning::
  If you are using a neural network with a topology that is substantially
  different to the FINN end-to-end examples, the simple dataflow build mode below
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
   Read more about the build configuration options on :py:mod:`finn.builder.build_dataflow_config.DataflowBuildConfig`.
   You can find an example .json file under ``src/finn/qnn-data/build_dataflow/dataflow_build_config.json``
4. (Optional) create a JSON file with the folding configuration. It must be named ``dataflow_build_dir/folding_config.json``.
   You can find an example .json file under ``src/finn/qnn-data/build_dataflow/folding_config.json``.
   Instead of specifying the folding configuration, you can use the `target_fps` option in the build configuration
   to control the degree of parallelization for your network.

Now you can invoke the simple dataflow build as follows:

::

  ./run-docker.sh build_dataflow <path/to/dataflow_build_dir/>

Depending on the chosen output products, the dataflow build will run for a while
as it goes through numerous steps:

.. code-block:: none

  Building dataflow accelerator from /home/maltanar/sandbox/build_dataflow/model.onnx
  Outputs will be generated at output_tfc_w1a1_Pynq-Z1
  Build log is at output_tfc_w1a1_Pynq-Z1/build_dataflow.log
  Running step: step_tidy_up [1/16]
  Running step: step_streamline [2/16]
  Running step: step_convert_to_hls [3/16]
  Running step: step_create_dataflow_partition [4/16]
  Running step: step_target_fps_parallelization [5/16]
  Running step: step_apply_folding_config [6/16]
  Running step: step_generate_estimate_reports [7/16]
  Running step: step_hls_codegen [8/16]
  Running step: step_hls_ipgen [9/16]
  Running step: step_set_fifo_depths [10/16]
  Running step: step_create_stitched_ip [11/16]
  Running step: step_measure_rtlsim_performance [12/16]
  Running step: step_make_pynq_driver [13/16]
  Running step: step_out_of_context_synthesis [14/16]
  Running step: step_synthesize_bitfile [15/16]
  Running step: step_deployment_package [16/16]


You can read a brief description of what each step does on
:py:mod:`finn.builder.build_dataflow_steps`. Note that a step whose output
products are not enabled will still run, but will do nothing.


Generated outputs
-----------------

.. note:: **All reports mentioned below are Python dictionaries exported as JSON.**

You will find the generated outputs under the subfolder you specified in the
build configuration, which can include a variety of folders and files
depending on the chosen output products.

The following outputs will be generated regardless of which particular outputs are selected:

* ``build_dataflow.log`` is the build logfile that will contain any warnings/errors
* ``time_per_step.json`` will report the time (in seconds) each build step took
* ``final_hw_config.json`` will contain the final (after parallelization, FIFO sizing etc) hardware configuration for the build
* ``intermediate_models/`` will contain the ONNX file(s) produced after each build step


The other output products are controlled by the `generate_outputs` field in the
build configuration), and are detailed below.

* :py:mod:`finn.builder.build_dataflow_config.DataflowOutputType.ESTIMATE_REPORTS` produces a variety of reports to estimate resource usage and performance *without* running any synthesis. This can be useful for setting up the parallelization and other hardware configuration:

  * ``report/estimate_layer_cycles.json`` -- cycles per layer estimation from analytical model
  * ``report/estimate_layer_resources.json`` -- resources per layer estimation from analytical model
  * ``report/estimate_layer_config_alternatives.json`` -- resources per layer estimation from analytical model, including what other config alternatives would have yielded
  * ``report/estimate_network_performance.json`` -- whole-network performance estimation from analytical model
  * ``report/op_and_param_counts.json`` -- per-layer and total number of operations and parameters (independent of parallelization)

* :py:mod:`finn.builder.build_dataflow_config.DataflowOutputType.STITCHED_IP`: produces a stitched Vivado IP block design that can be integrated with other FPGA designs in Vivado IPI:

  * ``stitched_ip/finn_vivado_stitch_proj.xpr`` -- Vivado project (including Vivado IP Integrator block design) to generate the stitched IP
  * ``stitched_ip/ip`` -- exported Vivado IP for the stitched design

* :py:mod:`finn.builder.build_dataflow_config.DataflowOutputType.RTLSIM_PERFORMANCE`: measure latency and performance for the stitched IP in RTL simulation, using PyVerilator

  * ``report/rtlsim_performance.json`` -- accelerator throughput and latency from RTL simulation

* :py:mod:`finn.builder.build_dataflow_config.DataflowOutputType.OOC_SYNTH` runs out-of-context synthesis for the stitched IP. This is useful for getting post-synthesis resource counts and achievable clock frequency without having to produce a full bitfile with DMA engines:

  * ``report/ooc_synth_and_timing.json`` -- resources and achievable clock frequency from out-of-context synthesis

* :py:mod:`finn.builder.build_dataflow_config.DataflowOutputType.BITFILE` will run Vivado and/or Vitis to insert the FINN accelerator inside a shell, with DMA engines instantiated to move data to/from main memory:

  * ``bitfile/finn-accel.(bit|xclbin)`` -- generated bitfile depending on platform
  * ``report/post_synth_resources.xml`` -- FPGA resource utilization after synthesis
  * ``report/post_route_timing.rpt`` -- post-route timing report


* :py:mod:`finn.builder.build_dataflow_config.DataflowOutputType.PYNQ_DRIVER` will generate a PYNQ Python driver that can be used to interface the generated accelerator:

  * ``driver/driver.py`` -- Python driver that can be used on PYNQ on Zynq or Alveo platforms to launch the accelerator

* :py:mod:`finn.builder.build_dataflow_config.DataflowOutputType.DEPLOYMENT_PACKAGE`:

  * ``deploy/`` -- deployment package folder with a bitfile and driver, ready to be copied to target hardware platform

Verification of intermediate steps
----------------------------------

FINN dataflow builds go through many steps before the bitfile is generated,
and the flow may produce erronous models due to bugs or unsupported features.
When running new models throught this process it's a good idea to enable the
verification features of the dataflow build. In this way, FINN will use the
input you provide to run through the intermediate models, produce some output
and compare it against the expected output that you provide.

This is achieved by setting up the following members of the build configuration:

* Set ``verify_steps`` to be a list of :py:mod:`finn.builder.build_dataflow_config.VerificationStepType`
  where each element in the list indicates the output of a particular step
  that will be verified. See the documentation of the ``VerificationStepType``
  for more information.
* Set ``verify_input_npy`` to the .npy filename to use as the test input to the
  verification process. We recommend using a single input example as the
  verification execution time can be lengthy for rtlsim, especially for larger
  networks. The shape of the numpy array must match the expected shape by
  the model.
* Set ``verify_expected_output_npy`` to the .npy filename to use as the "golden"
  output that the generated outputs will be compared against. The shape of the
  numpy array must match the produced output shape of the model.

The output of the verification is twofold:

* A message like ``Verification for folded_hls_cppsim : SUCCESS`` will appear in
  the build logfile.
* The output generated by the model at each verified step will be saved as a
  .npy file under ``verification_output/`` where each file created will indicate
  the verification step and the result of the verification (FAIL/SUCCESS).

Advanced mode
--------------

In other cases, you may want to have more control over the build process to
implement your own FINN flow with a different combination of compilation steps,
applying preprocessing to the model, calling custom transformations and so on.
This is possible by using the `build_custom` entry as follows:

1. Create a new folder for the custom build. It's best to keep this folder
outside the FINN repo folder for cleaner separation. Let's call this folder
``custom_build_dir``.

2. Create one or more Python files under this directory that perform the build(s)
you would like when executed, for instance ``custom_build_dir/build.py`` and
``custom_build_dir/build_quick.py``.
You should also put any ONNX model(s) or other
Python modules you may want to include in your build flow in this folder (so that they get
mounted into the Docker container while building). Besides the data placement,
you have complete freedom on how to implement the build flow here, including
calling the steps from the simple dataflow build mode above,
making calls to FINN library functions, preprocessing and altering models, building several variants etc.
You can find a basic example of a build flow under ``src/finn/qnn-data/build_dataflow/build.py``.

You can launch the desired custom build flow using:

::

 ./run-docker.sh build_custom <path/to/custom_build_dir> <name-of-build-flow>

This will mount the specified folder into the FINN Docker container and launch
the build flow. If ``<name-of-build-flow>`` is not specified it will default to ``build``
and thus execute ``build.py``. If it is specified, it will be ``<name-of-build-flow>.py``.
