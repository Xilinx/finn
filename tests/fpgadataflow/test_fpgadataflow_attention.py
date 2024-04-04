# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Testing framework
import pytest
# Use numpy for python execution / computing the ground truth expected values
import numpy as np

# Automatically generate init, repr, ... for classes containing a lot of
# attributes
from dataclasses import dataclass

# Utility types and function for creating onnx nodes and graphs
from onnx import TensorProto, helper

# QONNX datatypes
from qonnx.core.datatype import BaseDataType, DataType, FloatType, IntType
# Wrapper around ONNX model with some graph manipulation utility
from qonnx.core.modelwrapper import ModelWrapper
# Execute onnx model graphs
from qonnx.core.onnx_exec import execute_onnx
# Multithreshold activations
from qonnx.custom_op.general.multithreshold import multithreshold
# Registry of all QONNX CustomOps
from qonnx.custom_op.registry import getCustomOp
# Graph transformation giving unique names to each node in a QONNX model graph
from qonnx.transformation.general import GiveUniqueNodeNames
# QONNX utility for generating random input data for testing and for creating
# models
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

# Softmax function on numpy arrays with overflow handling matching the HLS
# operator
from finn.custom_op.fpgadataflow.attention import softmax
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim

# FINN graph transformations for preparing simulation (cppsim or rtlsim)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


# Python/Numpy model of the scaled dot-product attention operator as it is (will
# be...) implemented in the attention-hlslib
@dataclass
class MockScaledDotProductAttention:
    # Embedding dimension of queries and keys
    QKDim: int
    # Length of the query sequence
    QLen: int
    # Embedding dimension of the values
    VDim: int
    # Length of the key and value sequence
    KVLen: int

    # Folding along the embedding dimensions
    EmbFold: int
    # Folding along the sequence dimensions
    SeqFold: int

    # Datatype of query matrix elements
    QType: IntType
    # Datatype of key matrix elements
    KType: IntType
    # Datatype of value matrix elements
    VType: IntType
    # Datatype of mask matrix elements
    MType: IntType
    # Datatype of attention weights elements
    AType: IntType
    # Datatype of output elements
    OType: IntType

    # Datatype of accumulator elements of the Query x Key multiplication
    AccQKMatMul: IntType = DataType["UINT4"]
    # Datatype of output elements of the Query x Key multiplication
    OutQKMatMul: IntType = DataType["UINT4"]
    # Activation function type of the Query x Key multiplication
    ActQKMatMul: str = "thresholds"
    # Output bias to be applied to the thresholding activation following
    # the Query x Key multiplication
    BiasActQKMatMul: float = 0.0

    # Datatype of accumulator elements of the Attention x Value
    # multiplication
    AccAVMatMul: IntType = DataType["UINT4"]
    # Datatype of output elements of the Attention x Value
    # multiplication
    OutAVMatMul: IntType = DataType["UINT4"]
    # Activation function type of the Attention x Value multiplication
    ActAVMatMul: str = "thresholds"
    # Output bias to be applied to the thresholding activation following
    # the Attention x Value multiplication
    BiasActAVMatMul: float = 0.0

    # Scale factor preceding the softmax normalization to dequantize the
    # input
    DequantSoftmax: float = 1.0
    # Datatype of softmax normalization before applying activation or
    # type cast. THis is called Acc to stick to the naming scheme of the
    # MatMul operators before.
    #   Note: Currently this is ALWAYS floats
    AccASoftmax: FloatType = DataType["FLOAT32"]
    # Activation function type of the softmax normalization of the
    # attention weights
    ActASoftmax: str = "thresholds"
    # Output bias to be applied to the thresholding activation following
    # the softmax normalization of the attention weights
    BiasActASoftmax: float = 0.0

    # Initializes those parameters which depend on the initial configuration,
    # which is set by the generated __init__
    def __post_init__(self):
        # The last matmul output type must match with the specified output type
        assert self.OType == self.OutAVMatMul

        # Converts QONNX datatypes to their name (as a string)
        def maybe_name(value):
            # All QONNX datatypes are instances of the BaseDataType
            if isinstance(value, BaseDataType):
                # Convert to the name by referring to the datatypes name
                # attribute
                return value.name
            # Everything else is just assumed to be in the right format
            return value

        # Convert all node attributes which are registered so far to a
        # dictionary matching the CustomOp format, where DataTypes are converted
        # to string representations of their names
        self.node_attrs = {
            key: maybe_name(value) for key, value in self.__dict__.items()
        }

        # Dummy float type to use the threshold generator with flot inputs
        @dataclass
        class DummyFloat32:
            # Minimum and maximum of the represented float range
            _min: float
            _max: float

            # Getter for minimum of the represented range
            def min(self):
                return self._min

            # Getter for maximum of the represented range
            def max(self):
                return self._max

        # Generates thresholds representing a quantized identity function
        # mapping input datatype (idt) to output datatype (odt)
        def make_identity_thresholds(idt, odt, repeat=1):
            # The number of thresholds is determined by the range of the output
            # datatype
            steps = odt.get_num_possible_values() - 1
            # The scale, or step size, is determined by the ratio between input
            # and output range
            scale = (idt.max() - idt.min()) / (odt.max() - odt.min())
            # Generate step thresholds covering the input range and repeat for
            # multiple matrix rows/cols
            return np.array(
                repeat * [[scale * i + idt.min() for i in range(steps)]]
            ).astype(dtype=np.float32)

        # Generate identity function thresholds mapping the query-key matmul
        # accumulator type to the specified output type
        self.qk_thresholds = np.round(make_identity_thresholds(
            # Note: Repeat for all KVLen cols of the attention weights
            self.AccQKMatMul, self.OutQKMatMul, self.KVLen
        ))

        # Generate identity function thresholds mapping the float attention
        # weights to the specified integer type
        self.a_thresholds = make_identity_thresholds(
            # Note: Repeat for all KVLen cols of the attention weights
            DummyFloat32(0.0, 1.0), self.AType, self.KVLen
        )

        # Generate identity function thresholds mapping the attention-value
        # matmul accumulator type to the specified output type
        self.av_thresholds = np.round(make_identity_thresholds(
            # Note: Repeat for all VDim cols of the output
            self.AccAVMatMul, self.OutAVMatMul, self.VDim
        ))

    # Computes the query-key matmul with activation function simulating
    # quantization via thresholding
    def qk_matmul(self, query, key):
        return multithreshold(query @ key.T, self.qk_thresholds)

    # Computes the softmax normalization of attention weights with activation
    # function simulating quantization via thresholding
    def softmax(self, attention):
        # Input and output scale factors for float <-> int conversion
        iscale = self.DequantSoftmax
        # Scale the inputs, normalize using softmax and activate via thresholds
        return multithreshold(
            softmax(iscale * attention, axis=1), self.a_thresholds
        )

    # Computes the attention-value matmul with activation function simulating
    # quantization via thresholding
    def av_matmul(self, attention, value):
        return multithreshold(attention @ value, self.av_thresholds)

    # Computes scaled dot-product attention
    def __call__(self, query, key, value):
        return self.av_matmul(self.softmax(self.qk_matmul(query, key)), value)

    # Generates random sample inputs
    def make_rand_input(self):
        # Sample random query, key and value matrices with types and shapes
        # configured as attributes
        query = gen_finn_dt_tensor(self.QType, (self.QLen, self.QKDim))
        key = gen_finn_dt_tensor(self.KType, (self.KVLen, self.QKDim))
        value = gen_finn_dt_tensor(self.VType, (self.KVLen, self.VDim))
        # Return query, key, value tensors with integers represented as floats
        return query, key, value

    # Creates a QONNX ModelWrapper matching the attention configuration
    def make_modelwrapper(self):
        # Named threshold inputs
        #   Note: Order matters...
        thresholds = [
            "thresholds_qk_matmul",
            "thresholds_a_softmax",
            "thresholds_av_matmul",
        ]
        # Build up the node attribute dictionary
        kwargs = {
            # Refer to this operator type by its name
            "op_type": "ScaledDotProductAttention",
            # Execution will try to look up the implementation in the package
            # referred to by the domain
            "domain": "finn.custom_op.fpgadataflow",
            # Execution backend: Required attribute inherited from HLSCustomOp
            "backend": "fpgadataflow",
            # Named inputs and activation thresholds
            # TODO: Currently no masking support
            "inputs": ["Q", "K", "V", *thresholds],
            # Named model output
            "outputs": ["O"],
            # TODO: Currently no masking support
            "mask_mode": "none"
        }

        # Insert attributes into a new ONNX graph node
        node = helper.make_node(**kwargs, **self.node_attrs)

        # Create random sample inputs for shape inference
        q, k, v = self.make_rand_input()
        # Infer the output shape from the input shapes
        o_shape = (q.shape[0], v.shape[1])
        # Create onnx value info of all inputs and outputs assuming float
        # datatypes
        q_info = helper.make_tensor_value_info("Q", TensorProto.FLOAT, q.shape)
        k_info = helper.make_tensor_value_info("K", TensorProto.FLOAT, k.shape)
        v_info = helper.make_tensor_value_info("V", TensorProto.FLOAT, v.shape)
        o_info = helper.make_tensor_value_info("O", TensorProto.FLOAT, o_shape)
        # Collect input and output nodes in order
        inputs, outputs = [q_info, k_info, v_info], [o_info]

        # Create a graph connecting the scaled dot-product attention node to the
        # input and output nodes
        graph = helper.make_graph(
            [node], inputs=inputs, outputs=outputs, name='attention_graph'
        )
        # Wrap the ONNX graph in QONNX model wrapper
        model = ModelWrapper(qonnx_make_model(
            graph, producer_name='attention-model'
        ))

        # Add datatype annotations to all input tensors
        for tensor_name in kwargs["inputs"]:
            # Only annotate if a datatype is specified
            if f"{tensor_name}Type" in kwargs:
                # Update the datatype annotation
                model.set_tensor_datatype(
                    tensor_name, DataType[kwargs[f"{tensor_name}Type"]]
                )

        # Add datatype annotations to all output tensors
        for tensor_name in kwargs["outputs"]:
            # Only annotate if a datatype is specified
            if f"{tensor_name}Type" in kwargs:
                # Update the datatype annotation
                model.set_tensor_datatype(
                    tensor_name, DataType[kwargs[f"{tensor_name}Type"]]
                )

        # Set the threshold tensors as model initializer attributes of the
        # appropriate type
        #   TODO: Uses the actual input type to the multithreshold function as
        #    datatype. Somehow the mvau tests always use INT32, why?
        model.set_tensor_datatype("thresholds_qk_matmul", self.AccQKMatMul)
        model.set_initializer("thresholds_qk_matmul", self.qk_thresholds)

        model.set_tensor_datatype("thresholds_a_softmax", DataType["FLOAT32"])
        model.set_initializer("thresholds_a_softmax", self.a_thresholds)

        model.set_tensor_datatype("thresholds_av_matmul", self.AccAVMatMul)
        model.set_initializer("thresholds_av_matmul", self.av_thresholds)

        # Return the constructed qonnx model wrapper
        return model


# Size of query and key embedding dimension
@pytest.mark.parametrize("QKDim", [4, 8, 16])  # noqa: Duplicated code fragment
# Size of value embedding dimension
@pytest.mark.parametrize("VDim", [4, 8, 16])
# Length of key and value sequences
@pytest.mark.parametrize("KVLen", [16, 24])
# Length of query sequence
@pytest.mark.parametrize("QLen", [16, 24])
# Folding along the embedding dimensions
@pytest.mark.parametrize("EmbFold", [2])
# Folding along the sequence dimensions
@pytest.mark.parametrize("SeqFold", [8])
# Datatypes of queries, keys and values, mask and output
@pytest.mark.parametrize("QType", [DataType["UINT8"]])
@pytest.mark.parametrize("KType", [DataType["UINT8"]])
@pytest.mark.parametrize("VType", [DataType["UINT8"]])
@pytest.mark.parametrize("MType", [DataType["UINT8"]])
@pytest.mark.parametrize("AType", [DataType["UINT8"]])
@pytest.mark.parametrize("OType", [DataType["UINT8"]])
# Different modes to provide a mask
@pytest.mark.parametrize("mask", ["none"])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
# Tests cpp simulation of single scaled dot-product attention head
def test_attention_cppsim(
        # Shape configuration
        QKDim,  # noqa: "Argument should be lowercase"
        VDim,  # noqa
        KVLen,  # noqa
        QLen,  # noqa
        # Folding configuration
        EmbFold,  # noqa
        SeqFold,  # noqa
        # Type configuration
        QType,  # noqa
        KType,  # noqa
        VType,  # noqa
        MType,  # noqa
        AType,  # noqa
        OType,  # noqa
        # Type of mask to use: either 'none', 'input', or 'causal'
        mask
):
    # Attention instance simulating in python and generating a matching
    # QONNX configuration
    attention = MockScaledDotProductAttention(  # noqa: Duplicated code fragment
        # Shape configuration
        QKDim=QKDim,
        QLen=QLen,
        VDim=VDim,
        KVLen=KVLen,
        # Folding configuration
        EmbFold=EmbFold,
        SeqFold=SeqFold,
        # Type configuration
        QType=QType,
        KType=KType,
        VType=VType,
        MType=MType,
        AType=AType,
        OType=OType,
        # Accumulator type configuration
        AccQKMatMul=DataType["UINT32"],
        OutQKMatMul=DataType["UINT8"],
        AccAVMatMul=DataType["UINT32"],
        OutAVMatMul=OType,
        # Dequantizer scale, factor to convert the whole UINT8 range to floats
        # in range 0.0 to 1.0
        DequantSoftmax=1.0 / (DataType["UINT8"].get_num_possible_values() - 1)
    )

    # Create a QONNX model wrapper for testing
    model = attention.make_modelwrapper()
    # Sample some random inputs
    q, k, v = attention.make_rand_input()
    # Prepare execution context
    context = {
        "Q": q, "K": k, "V": v, "mask": mask
    }

    # Mark all nodes to be specialized as HLS backend implementations
    for node in model.graph.node:
        # Get the CustomOp instance of the node to get access to the node
        # attributes
        inst = getCustomOp(node)
        # Note: only HLS-based layers execute C++ Simulation
        inst.set_nodeattr("preferred_impl_style", "hls")
    # Turn all HWCustomOp layers into HLS specializations
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))

    # Set model execution mode to C++ simulation
    model = model.transform(SetExecMode("cppsim"))
    # Generates the C++ source and compiles the C++ simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    # Compute ground-truth output in software
    o_expected = attention(q, k, v)  # noqa: Duplicated code fragment
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["O"]

    # Log outputs for debugging
    print(f"{o_expected}\n", file=open('o_expected_cppsim.txt', 'w'))
    print(f"{o_produced}\n", file=open('o_produced_cppsim.txt', 'w'))
    # Save the ONNX model graph for debugging
    model.save("attention-cppsim.onnx")

    # Test whether the expectation and the onnx model output match
    assert np.allclose(o_produced, o_expected), "cppsim exec failed"


# Size of query and key embedding dimension
@pytest.mark.parametrize("QKDim", [4])  # noqa: Duplicated code fragment
# Size of value embedding dimension
@pytest.mark.parametrize("VDim", [4])
# Length of key and value sequences
@pytest.mark.parametrize("KVLen", [16])
# Length of query sequence
@pytest.mark.parametrize("QLen", [16])
# Folding along the embedding dimensions
@pytest.mark.parametrize("EmbFold", [2])
# Folding along the sequence dimensions
@pytest.mark.parametrize("SeqFold", [8])
# Datatypes of queries, keys and values, mask and output
@pytest.mark.parametrize("QType", [DataType["UINT8"]])
@pytest.mark.parametrize("KType", [DataType["UINT8"]])
@pytest.mark.parametrize("VType", [DataType["UINT8"]])
@pytest.mark.parametrize("MType", [DataType["UINT8"]])
@pytest.mark.parametrize("AType", [DataType["UINT8"]])
@pytest.mark.parametrize("OType", [DataType["UINT8"]])
# Different modes to provide a mask
@pytest.mark.parametrize("mask", ["none"])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
# Tests rtl simulation of single scaled dot-product attention head
def test_attention_rtlsim(
        # Shape configuration
        QKDim,  # noqa: "Argument should be lowercase"
        VDim,  # noqa
        KVLen,  # noqa
        QLen,  # noqa
        # Folding configuration
        EmbFold,  # noqa
        SeqFold,  # noqa
        # Type configuration
        QType,  # noqa
        KType,  # noqa
        VType,  # noqa
        MType,  # noqa
        AType,  # noqa
        OType,  # noqa
        # Type of mask to use: either 'none', 'input', or 'causal'
        mask
):
    # Attention instance simulating in python and generating a matching
    # QONNX configuration
    attention = MockScaledDotProductAttention(  # noqa: Duplicated code fragment
        # Shape configuration
        QKDim=QKDim,
        QLen=QLen,
        VDim=VDim,
        KVLen=KVLen,
        # Folding configuration
        EmbFold=EmbFold,
        SeqFold=SeqFold,
        # Type configuration
        QType=QType,
        KType=KType,
        VType=VType,
        MType=MType,
        AType=AType,
        OType=OType,
        # Accumulator type configuration
        AccQKMatMul=DataType["UINT32"],
        OutQKMatMul=DataType["UINT8"],
        AccAVMatMul=DataType["UINT32"],
        OutAVMatMul=OType,
        # Dequantizer scale, factor to convert the whole UINT8 range to floats
        # in range 0.0 to 1.0
        DequantSoftmax=1.0 / (DataType["UINT8"].get_num_possible_values() - 1)
    )

    # Create a QONNX model wrapper for testing
    model = attention.make_modelwrapper()
    # Sample some random inputs
    q, k, v = attention.make_rand_input()
    # Prepare execution context
    context = {
        "Q": q, "K": k, "V": v, "mask": mask
    }

    # Mark all nodes to be specialized as HLS backend implementations
    for node in model.graph.node:
        # Get the CustomOp instance of the node to get access to the node
        # attributes
        inst = getCustomOp(node)
        # Note: only HLS-based layers execute C++ Simulation
        inst.set_nodeattr("preferred_impl_style", "hls")
    # Turn all HWCustomOp layers into HLS specializations
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))

    # Set model execution mode to RTL simulation
    model = model.transform(SetExecMode("rtlsim"))
    # Generates the C++ source and compiles the RTL simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    # Compute ground-truth output in software
    o_expected = attention(q, k, v)  # noqa: Duplicated code fragment
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["O"]

    # Log outputs for debugging
    print(f"{o_expected}\n", file=open('o_expected_rtlsim.txt', 'w'))
    print(f"{o_produced}\n", file=open('o_produced_rtlsim.txt', 'w'))
    # Save the ONNX model graph for debugging
    model.save("attention-rtlsim.onnx")

    # Test whether the expectation and the onnx model output match
    assert np.allclose(o_produced, o_expected), "rtlsim exec failed"
