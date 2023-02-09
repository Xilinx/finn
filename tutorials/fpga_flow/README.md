# FINN Example FPGA Flow Using MNIST Numerals

This example demonstrates how to bring a FINN compiled model into the Vivado FPGA design environment for integration into a larger FPGA application. It extends on the command-line [build_dataflow](https://github.com/Xilinx/finn/tree/master/src/finn/qnn-data/build_dataflow) using a model that was quantized with [Brevitas](https://github.com/Xilinx/brevitas) down to single-bit weight/ativation precision to classify hand-written numerals from the MNIST data set.

If you are new to the command-line flow, more information can be found [here](https://finn.readthedocs.io/en/latest/command_line.html).

This demo was created using Vivado 2022.1.

## Compiling the Model in FINN

#### Configuration
`build.py` assembles the needed files and configures how the model is compiled when generating the "stitched IP".  The following items will need to be set appropriately for specific use cases:
- `output_dir`: defines the directory to be created for FINN compiler output.
- `target_fps`: desired throughput performance target for FINN compiler to achieve.
- `mvau_wwidth_max`:  _an optional parameter_ ([described here](https://finn.readthedocs.io/en/latest/source_code/finn.builder.html#finn.builder.build_dataflow_config.DataflowBuildConfig.mvau_wwidth_max)) shown only to illustrate passing additional configuration items to the compiler.
- `folding_config_file`: an optional parameter to pass a json file defining the layer optimizations (PE,SIMD,ramstyle, etc.) to the compiler.
- `synth_clk_period_ns`: set the desired clock period in nS.
- `fpga_part` configures the IP for your target device that the stitched IP will be implemented in.  It should be the full string recognized in Vivado: \<device\>-\<package\>-\<temperature_grade\>-\<speed_grade\>
- `generate_outputs`: for integration purposes, the only output needed is `STITCHED_IP`.  You might also find the `ESTIMATE_REPORTS` interesting.  Other options are documented [here](https://finn.readthedocs.io/en/latest/command_line.html#generated-outputs) and some of them (namely OOC_SYNTH, BITFILE) add substantial runtime and are not needed for this flow.
- `stitched_ip_gen_dcp` : will generate an IP block with a synthesized design checkpoint (.dcp) which makes the design more portable across different machines, but will add some runtime.


### Running FINN Compiler

Prior to running, insure the following prerequisites have been met:
- Install FINN and prerequisites.  The [Getting Started](https://finn.readthedocs.io/en/latest/getting_started.html#quickstart) section of the FINN documentation might be helpful for this.
- Ensure you have the `FINN_XILINX_PATH` and `FINN_XILINX_VERSION` env variables set appropriately for your install.  For example:
> export FINN_XILINX_PATH=/opt/Xilinx
> export FINN_XILINX_VERSION=2022.1
- Set the env variable for your `finn` install top directory (where you cloned the FINN compiler repo):
> export FINN_ROOT=/home/foo/finn

Then, change to `finn` install directory and invoke the build as follows:
> cd ${FINN_ROOT}
> ./run-docker.sh build_custom ${FINN_ROOT}/tutorials/fpga_flow/

Alternatively, since the tutorials folder is already part of the FINN compiler installation, you can invoke it from within the Docker container:
> cd ${FINN_ROOT}
> ./run-docker.sh
> cd tutorials/fpga_flow
> python build.py

The build should finish in about 10 minutes, and the FINN docker will close on success.

```
   ...
   Running step: step_create_stitched_ip [12/18]
   Running step: step_measure_rtlsim_performance [13/18]
   Running step: step_out_of_context_synthesis [14/18]
   Running step: step_synthesize_bitfile [15/18]
   Running step: step_make_pynq_driver [16/18]
   Running step: step_deployment_package [17/18]
   Running step: custom_step_gen_tb_and_io [18/18]
   Completed successfully
   The program finished and will be restarted
```


### Examine the Stitched IP

Navigate to the stitched IP project directory:

> cd ${FINN_ROOT}/tutorials/fpga_flow/output_tfc_w0a1_fpga/stitched_ip

And, open the project:

> vivado finn_vivado_stitch_proj.xpr

Explore the IPI board design and note the interfaces.


### Simulating the Stitched IP with a Verilog Test Bench

You may have noticed that the final build step invoked by FINN is `custom_step_gen_tb_and_io`.
This custom step generates the files we'll need to simulate the FINN design in Vivado, and places
them under `${FINN_ROOT}/tutorials/fpga_flow/output_tfc_w0a1_fpga/sim`. Let's examine these files.

* `input.dat` and `expected_output.dat`: text files containing hex data for sample input and its expected
   output. These are generated from the `input.npy` and `expected_output.npy` files by the FINN compiler.
   Notice how the structure of the .dat files reflects the parallelization parameters of the first (for input)
   and last (for output) layers of the hardware. The input is fed 49 bytes at a time, over 19 cycles to finish
   a sample of 28x28=784 bytes from the MNIST dataset. Note how this matches PE=49 as selected for the first layer in `folding_config.json`. Additionally, note the reversal along each line in the .dat file to align the
   byte order with what the FINN-generated hardware expects.

* `finn_testbench.sv` : created by filling in a testbench template (under `templates/finn_testbench.template.sv`) with
   relevant information by the FINN compiler, including the sizes of the input/output streams, folding factors and number of samples in the generated .dat file.

* `make_sim_proj.tcl` : created by filling in a TCL script template (under `templates/make_sim_proj.template.tcl`) by
   the FINN compiler. Used for launching the testbench simulation.

You can now launch the simulation as follows:

> cd ${FINN_ROOT}/tutorials/fpga_flow/output_tfc_w0a1_fpga/sim
> vivado -mode gui -source make_sim_proj.tcl

The simulation should complete with:

```
 # run all
CHK: Data    match 02 == 02   --> 0

************************************************************
  SIM COMPLETE
  Validated 1 data points
  Total error count: ====>  0  <====
```

You can also use the provided testbench skeleton and the custom step in `build.py` to build your own
testbench generators.

#### Instantiation in Mission Design

There are any number of ways to bring the stitched IP into larger design.

FINN already packages the stitched IP block design as a standalone IP-XACT component, which you can find under `${FINN_ROOT}/tutorials/fpga_flow/output_tfc_w0a1_fpga/stitched_ip/ip`. You can add this to the list of IP repos and use it in your own Vivado designs. A good reference for this is [UG1119](https://www.xilinx.com/content/dam/xilinx/support/documents/sw_manuals/xilinx2022_1/ug1119-vivado-creating-packaging-ip-tutorial.pdf)

Keep in mind that all of the User IP Repo's included in the Stitched IP project (from `$FINN_HOST_BUILD_DIR` which is normally located under `/tmp/finn_dev_<username>`) need to also be brought in as IP Repo's to any project using the stitched IP.  It would be prudent to copy those IP repos to an appropriate archive location. You should also set the
`FINN_ROOT` environment variable to point to the compiler installation directory, as some of the build scripts will
use this to access various components. Alternatively, if you don't want to copy all of the dependencies, you can ask FINN to generate the IP-XACT component with a synthesized .dcp checkpoint by passing the [stitched_ip_gen_dcp=True](https://finn-dev.readthedocs.io/en/latest/source_code/finn.builder.html#finn.builder.build_dataflow_config.DataflowBuildConfig.stitched_ip_gen_dcp) option as part of the build configuration.
