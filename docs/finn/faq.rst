.. _faq:

***************************
Frequently Asked Questions
***************************

Can't find the answer to your question here? Check `FINN GitHub Discussions <https://github.com/Xilinx/finn/discussions>`_.


Can I install FINN out of the Docker container?
    We do not support out of the Docker implementations at the moment. This is due
    to the high complexity of the FINN project dependencies.

Since FINN uses ONNX, can I compile any model from the ONNX Model Zoo to an FPGA accelerator?
    The short answer is no. FINN uses ONNX in a specific (non-standard) way, including custom layer
    types and quantization annotations. Networks must be first quantized using Brevitas and exported
    to FINN-ONNX to be converted to FPGA accelerators.


Can I deploy custom NNs with arbitrary precisions and layers using FINN?
    Yes, though the effort required and quality of results will vary.
    Although we do support arbitrary
    precision, the way we create the hardware isn't typically practical for more than
    4 bits, or very large networks for a single FPGA.
    In terms of layers, only a subset of quantized layers covered by the various FINN examples
    are currently supported.
    It is possible to add support for new layers, though we don't have tutorials for this in place
    just yet.

Does FINN only work with the example networks?
    FINN isn't restricted to the example networks;
    rather, it's restricted to certain patterns (e.g. certain layer types and their combinations).
    The current best practice for custom networks is to take a working network and gradually modify it.

What is the expected background for using FINN?
    Some general knowledge of Python, Docker, machine learning with neural networks and Jupyter notebooks
    is expected.
    Our goal is to make the tool in a shape and form so that no hardware/FPGA background
    should be necessary, although having some knowledge would give better results.

What operating systems are supported by FINN?
    FINN should work fine under any Linux-based OS capable of running Vivado/Vitis, as long
    as you install Docker (``docker-ce``) on your machine.

I am getting DocNav and Model_Composer errors when launching the Docker image.
    We do not mount those particular directories into the Docker container because they are not
    used. The errors are Vivado related but you can safely ignore them.

What board do you recommend to start working with FINN?
    Our preferred target platforms are those supported by  `PYNQ <http://www.pynq.io/board.html>`_.
    For those boards we can offer end-to-end (DNN-to-bitstream) deployment,
    see the `finn-examples <https://github.com/Xilinx/finn-examples>`_ repository for some examples.
    However, FINN also supports Vivado IP Integrator designs. The IPs connect using AXI stream (FIFO)
    in-and-out interfaces. This means that it can be integrated onto any Xilinx FPGA board,
    though you will have to do the system integration manually.

FINN-generated builds break after I restart my computer, because ``/tmp`` gets wiped.
    See https://github.com/Xilinx/finn/discussions/404

How can I target an arbitrary Xilinx FPGA without PYNQ support?
    See https://github.com/Xilinx/finn/discussions/387

Why does FINN-generated architectures need FIFOs between layers?
    See https://github.com/Xilinx/finn/discussions/383

How do I tell FINN to utilize a particular type of memory resource in particular layers?
    This is controlled by the ``ram_style`` attribute for most hardware layers and ``depth_trigger_bram``/``depth_trigger_uram`` for RTL Thresholding.
    See :ref:`mem_mode` for detailed information about memory modes (internal_embedded, internal_decoupled, external) and
    how to control memory primitive selection (LUTRAM, BRAM, URAM) for different layer types.
    When using ``build_dataflow``, these can be specified per layer in the folding config file
    (:py:mod:`finn.builder.build_dataflow_config.DataflowBuildConfig.folding_config_file`).
    See the `MobileNet-v1 build config for ZCU104 in finn-examples <https://github.com/Xilinx/finn-examples/blob/main/build/mobilenet-v1/folding_config/ZCU104_folding_config.json#L15>`_ for reference.

Which data layout do FINN-generated accelerators use? Big-endian? Little-endian?
    The data layout used by FINN does not correspond to system-level big or little endian due to difficulties in defining what
    the “word size” is and bit packing for smaller datatypes. FINN’s “word size” is dependent on the parallelization of the
    first/last layers. For instance, if the first HLS layer is using SIMD=3 this means the “innermost dimension” in the
    data packing functions will be of size 3.
    When you use the verification infrastructure or the generated PYNQ Python drivers that FINN provides, the tool normally
    takes care of any required data layout conversion on standard numpy arrays before presenting the data to the accelerator,
    and vice versa on the output side. Doing this data packing and layout conversion manually can be messy at the moment.
    If you need to do this manually, first examine how the `FINN PYNQ Python drivers <https://github.com/Xilinx/finn-examples/blob/main/finn_examples/driver.py#L379>`_ do this – notice how the input data is
    first reshaped to create the “folded input shape” that reflects the word size of the first layer based on how much it
    was parallelized, then data packing is applied to obtain a raw byte array (with some reversals going on) that can be
    fed directly to the hardware. Another example of this is the `npy_to_rtlsim_input <https://github.com/Xilinx/finn/blob/dev/src/finn/util/data_packing.py#L284>`_ function, which converts npy arrays to lists of Python arbitrary-precision integers that we feed into xsi for rtl simulation.

Why does FIFO sizing take so long for my network? Is something wrong?
    The automatic FIFO sizing in FINN can take quite long. It unfortunately doesn’t really parallelize on multiple cores since
    it’s based on running an rtl simulation with lots of inputs and very large FIFOs, then observing the max occupancy/count
    in each FIFO.

What's a good starting point for the folding configuration if I want to make manual changes?
    First, enable automatic folding options in ``build_dataflow`` such ``target_fps``. This should find a decent set of
    folding factors and save them to ``output_folder/auto_folding_config.json`` which you can use as a basis for creating the desired config.

How do I reduce resource utilization for my model?
    High resource usage typically comes from high parallelization (PE/SIMD values). To reduce resources:

    1. Lower PE and SIMD values in your folding configuration - this trades throughput for lower resource usage
    2. Use ``internal_decoupled`` mem_mode with appropriate ``ram_style`` to better control memory primitive usage (see :ref:`mem_mode`)
    3. Check the ``estimate_layer_resources.json`` report to identify which layers consume the most resources

How do I configure external mem_mode for weights?
    External mem_mode streams weights from external DRAM at runtime rather than embedding them on-chip. This is useful when:

    - You want to change weights without regenerating the bitstream
    - You have very large weight tensors that don't fit in on-chip memory
    - You need to support multiple models with the same accelerator hardware

    Configure external mode by setting ``"mem_mode": "external"`` for specific layers in your folding configuration file:

    .. code-block:: json

        {
            "MVAU_hls_1": {
                "PE": 16,
                "SIMD": 16,
                "mem_mode": "external"
            }
        }

    The compiler creates IODMA (Input/Output DMA) nodes to stream weights from external memory. Runtime weights are saved
    as ``idma{name}.npy`` files in the deployment package's ``runtime_weights/`` directory. See ``tests/end2end/test_ext_weights.py``
    for a complete example.

What quantization approaches work best with FINN?
    FINN works best with:

    - Integer quantization from 1-bit (binary/bipolar) to 8-bit is recommended. FINN supports arbitrary-precision integer bitwidths (e.g., int16), but keep in mind that higher bitwidths will consume more FPGA resources.
    - Symmetric quantization for weights (zero-point = 0)
    - Per-tensor and per-channel quantization (both supported)
    - Quantization-aware training (QAT) and post-training quantization (PTQ) via Brevitas

    See the `QAT guidelines <https://bit.ly/finn-hls4ml-qat-guidelines>`_ for detailed recommendations.
    Export your trained model to QONNX (supports both QAT and PTQ), then convert to FINN-ONNX following the `Brevitas network import tutorial <https://github.com/Xilinx/finn/blob/main/notebooks/basics/1_brevitas_network_import_via_QONNX.ipynb>`_.

Can I use FINN for models with custom layer types not in the examples?
    Yes, but you'll need to implement the layer yourself. FINN supports adding new hardware layers by:

    1. Creating a base layer class (backend-agnostic) in ``src/finn/custom_op/fpgadataflow/``
    2. Optionally implementing HLS variant in ``src/finn/custom_op/fpgadataflow/hls/``
    3. Optionally implementing RTL variant in ``src/finn/custom_op/fpgadataflow/rtl/``
    4. Adding corresponding templates to finn-hlslib or finn-rtllib

    See :doc:`/implementation/index` for detailed guidance on extending FINN with new operators.
