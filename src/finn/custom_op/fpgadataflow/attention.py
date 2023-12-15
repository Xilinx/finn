# Operating system stuff, e.g. paths
import os
# Python warning subsystem
import warnings
# Numpy math and arrays
import numpy as np

# Derive custom operators form the FINN base custom op
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
# Convert and pack (numpy) data for C++ code generation
from finn.util.data_packing import numpy_to_hls_code
# QONNX/FINN datatypes
from qonnx.core.datatype import DataType  # noqa qonnx dependency is specified
# in setup.cfg as well as in fetch-repos.sh
# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper  # noqa
# Partitions tensor into folded/pe groups
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions  # noqa


# Softmax function on numpy arrays with overflow handling matching the HLS
# operator
def softmax(x, axis):
    # For overflow handling, find the maximum value along axis and place ones at
    # each occurrence
    max_ones = (x == np.max(x, axis=axis, keepdims=True)).astype(np.float32)
    # Count the occurrences of the maximum along the normalization axis
    max_counts = np.sum(max_ones, axis=axis, keepdims=True)
    # Exponential of the input
    exp = np.exp(x)
    # Compute the total along axis
    total = np.sum(exp, axis=axis, keepdims=True)
    # Detect overflow of the summation
    overflow = np.isinf(total)
    # Replace overflows by equal weight given to all instances of the maximum
    # input value. For non overflow just compute normal softmax
    return np.where(overflow, max_ones / max_counts, exp / total)


# Scaled Dot-Product Attention Custom Operator
#   Note: Single head attention
class ScaledDotProductAttention(HLSCustomOp):
    # Initializes the operator given an onnx graph node
    def __init__(self, onnx_node, **kwargs):
        # Just forward all arguments to the init method of the CustomOp base
        super().__init__(onnx_node, **kwargs)

    # WIP: Refactor the node attributes matching the HLS operator which is WIP
    # in another repository right now.
    def get_nodeattr_types(self):
        # Start from parent operator class attributes
        attrs = super().get_nodeattr_types()
        # Update attributes dictionary for new custom operator
        attrs.update({
            # Embedding dimension of queries and keys
            "QKDim": ("i", True, 0),
            # Length of the query sequence
            "QLen": ("i", True, 0),
            # Embedding dimension of the values
            "VDim": ("i", True, 0),
            # Length of the key and value sequence
            "KVLen": ("i", True, 0),

            # Folding along the embedding dimensions
            "EmbFold": ("i", True, 0),
            # Folding along the sequence dimensions
            "SeqFold": ("i", True, 0),

            # Datatype of query matrix elements
            "QType": ("s", True, ""),
            # Datatype of key matrix elements
            "KType": ("s", True, ""),
            # Datatype of value matrix elements
            "VType": ("s", True, ""),
            # Datatype of mask matrix elements
            "MType": ("s", False, "INT0"),
            # Datatype of attention weights elements
            "AType": ("s", False, "UINT32"),
            # Datatype of output elements
            "OType": ("s", True, ""),

            # Datatype of accumulator elements of the Query x Key multiplication
            "AccQKMatMul": ("s", False, "UINT32"),
            # Datatype of output elements of the Query x Key multiplication
            "OutQKMatMul": ("s", False, "UINT32"),
            # Activation function type of the Query x Key multiplication
            "ActQKMatMul": ("s", False, "none", {"none", "thresholds"}),
            # Output bias to be applied to the thresholding activation following
            # the Query x Key multiplication
            "BiasActQKMatMul": ("f", False, 0.0),

            # Datatype of accumulator elements of the Attention x Value
            # multiplication
            "AccAVMatMul": ("s", False, "UINT32"),
            # Datatype of output elements of the Attention x Value
            # multiplication
            "OutAVMatMul": ("s", False, "UINT32"),
            # Activation function type of the Attention x Value multiplication
            "ActAVMatMul": ("s", False, "none", {"none", "thresholds"}),
            # Output bias to be applied to the thresholding activation following
            # the Attention x Value multiplication
            "BiasActAVMatMul": ("f", False, 0.0),


            # Scale factor preceding the softmax normalization to dequantize the
            # input
            "DequantSoftmax": ("f", False, 1.0),
            # Datatype of softmax normalization before applying activation or
            # type cast. THis is called Acc to stick to the naming scheme of the
            # MatMul operators before.
            #   Note: Currently this is ALWAYS floats
            "AccASoftmax": ("s", False, "FLOAT32"),
            # Activation function type of the softmax normalization of the
            # attention weights
            "ActASoftmax": ("s", False, "none", {"none", "thresholds"}),
            # Output bias to be applied to the thresholding activation following
            # the softmax normalization of the attention weights
            "BiasActASoftmax": ("f", False, 0.0),

            # Mode used for providing the attention mask: There can be no mask,
            # a mask sent as the fourth dynamic input, a mask provided as fourth
            # constant input or a causal attention mask which is generated by
            # the operator itself.
            "mask_mode": (
                "s", True, "none", {"none", "input", "const", "causal"}
            ),

            # Possible execution modes for simulating this node
            #   Note: Override to support python mode
            "exec_mode": (
                "s", False, "python", {"", "rtlsim", "cppsim", "python"}
            ),

            # Input and output FIFO depths for multi-I/O nodes
            #   Note: Need to override here as there are three inputs
            "inFIFODepths": ("ints", False, [2, 2, 2]),
            "outFIFODepths": ("ints", False, [2]),
        })
        # Return updated attribute dictionary
        return attrs

    # Shape configuration of the operator
    @property
    def shapes(self):
        # Note: This matches the order of definition above and the order of the
        # HLS lib template as well
        return (self.get_nodeattr("QKDim"), self.get_nodeattr("QLen"),
                self.get_nodeattr("VDim"), self.get_nodeattr("KVLen"))

    # Folding configuration of the operator
    @property
    def folds(self):
        # Note: This matches the order of definition above and the order of the
        # HLS lib template as well
        return self.get_nodeattr("EmbFold"), self.get_nodeattr("SeqFold")

    # Tests whether the given folding is a valid configuration with respect to
    # the shape configuration
    @property
    def is_valid_folding(self):
        # Get and unpack the shape attributes (except the q matrix length, which
        # is never folded)
        qkdim, _, vdim, kvlen = self.shapes
        # Get and unpack the folding attributes
        embfold, seqfold = self.folds
        # All shapes must be multiples of their corresponding fold
        return not ((qkdim % embfold) or (vdim % embfold) or (kvlen % seqfold))

    # Returns an ONNX node that has the same shape inference behavior
    def make_shape_compatible_op(self, model):
        # Infer the output shape from the input shapes
        o_shape = (self.get_nodeattr("QLen"), self.get_nodeattr("VDim"))
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the shape of the input tensor for inferring the number of
        # heads and correctly propagating shapes
        shape = model.get_tensor_shape(node.input[0])
        # Determine the rank of the input tensor to support batched and
        # non-batched inputs
        rank = len(shape)
        # Constant operation producing output of given shape
        #   Note: Rank == 3 allows for propagating yet unrolled multi-attention
        #   heads.
        return super().make_const_shape_op(
            (shape[0], *o_shape) if (rank == 3) else o_shape
        )

    # Infers the output data types and updates the input datatypes of the node
    def infer_node_datatype(self, model):
        # ONNX graph node of the operator
        node = self.onnx_node

        # Get input datatypes from model for query, key, value nodes in order
        q_dtype = model.get_tensor_datatype(node.input[0])
        k_dtype = model.get_tensor_datatype(node.input[1])
        v_dtype = model.get_tensor_datatype(node.input[2])

        # Test for changing query input datatype
        if q_dtype != self.get_nodeattr("QType"):
            # Issue a warning message
            warnings.warn("QType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_nodeattr("QType")),
                str(q_dtype),
            ))
        # Test for changing key input datatype
        if k_dtype != self.get_nodeattr("KType"):
            # Issue a warning message
            warnings.warn("KType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_nodeattr("KType")),
                str(k_dtype),
            ))
        # Test for changing value input datatype
        if v_dtype != self.get_nodeattr("VType"):
            # Issue a warning message
            warnings.warn("VType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_nodeattr("VType")),
                str(v_dtype),
            ))

        # Update the node datatype attributes
        self.set_nodeattr("QType", q_dtype.name)
        self.set_nodeattr("KType", k_dtype.name)
        self.set_nodeattr("VType", v_dtype.name)

        # Attention mask might be provided as an input as well
        if self.get_nodeattr("mask_mode") == "input":
            # Get the datatype attribute of the attention mask
            #   Note: Optional mask will be provided as the fourth input
            mask_dtype = model.get_tensor_datatype(node.input[3])
            # Test for changing mask input datatype
            if mask_dtype != self.get_nodeattr("MType"):
                # Issue a warning message
                warnings.warn("MType changing for %s: %s -> %s " % (
                    node.name,
                    str(self.get_nodeattr("MType")),
                    str(mask_dtype),
                ))
            # Update the node datatype attribute of the attention mask
            self.set_nodeattr("MType", mask_dtype.name)

        # Set the model output datatype
        model.set_tensor_datatype(
            node.output[0], DataType[self.get_nodeattr('OType')]
        )

    # Executes the attention operator in python mode simulation
    def _execute_node_python(self, context, graph):  # noqa: graph unused
        # Multithreshold activations
        from qonnx.custom_op.general.multithreshold import multithreshold # noqa

        # Get the node wrapped by this custom op
        node = self.onnx_node

        # Read the input from the execution context and reshape to match the
        # expected folding
        q = context[node.input[0]].reshape(self.get_normal_input_shape(ind=0))
        k = context[node.input[1]].reshape(self.get_normal_input_shape(ind=1))
        v = context[node.input[2]].reshape(self.get_normal_input_shape(ind=2))

        # Quantization activation function following the query and key
        # multiplication
        def act_qk_matmul(x):
            # Only applies if this is specified as a thresholding activation
            if self.get_nodeattr("ActQKMatMul") == "thresholds":
                # Get the thresholds initializer by name from ordered list of
                # optional inputs
                thresholds = context[
                    self.get_input_name_by_name("thresholds_qk_matmul")
                ]
                # Activation value, i.e., bias applied after thresholding
                # activation
                bias = self.get_nodeattr("BiasActQKMatMul")
                # Applies thresholding activation in python to the input
                return multithreshold(x, thresholds) + bias
            # If not thresholds, assume identity function
            return x

        # Quantization activation function following the softmax normalization
        def act_a_softmax(x):
            # Only applies if this is specified as a thresholding activation
            if self.get_nodeattr("ActASoftmax") == "thresholds":
                # Get the thresholds initializer by name from ordered list of
                # optional inputs
                thresholds = context[
                    self.get_input_name_by_name("thresholds_a_softmax")
                ]
                # Activation value, i.e., bias applied after thresholding
                # activation
                bias = self.get_nodeattr("BiasActASoftmax")
                # Applies thresholding activation in python to the input
                return multithreshold(x, thresholds) + bias
            # If not thresholds, assume identity function
            return x

        # Quantization activation function following the attention and values
        # multiplication
        def act_av_matmul(x):
            # Only applies if this is specified as a thresholding activation
            if self.get_nodeattr("ActAVMatMul") == "thresholds":
                # Get the thresholds initializer by name from ordered list of
                # optional inputs
                thresholds = context[
                    self.get_input_name_by_name("thresholds_av_matmul")
                ]
                # Activation value, i.e., bias applied after thresholding
                # activation
                bias = self.get_nodeattr("BiasActAVMatMul")
                # Applies thresholding activation in python to the input
                return multithreshold(x, thresholds) + bias
            # If not thresholds, assume identity function
            return x

        # Scale used to dequantize the qk matrix before computing the softmax in
        # floating point
        dequant = self.get_nodeattr("DequantSoftmax")

        # 1. Queries and keys multiplication followed by quantizing activation
        # function
        qk = act_qk_matmul(np.matmul(q, k.T))
        # Softmax-normalization of the attention weights followed by quantizing
        # activation function
        a = act_a_softmax(softmax(dequant * qk, axis=1))
        # 2. Attention weights and values matmul followed by quantization
        # activation function
        out = act_av_matmul(np.matmul(a, v))

        # Insert the results into the execution context
        context[self.onnx_node.output[0]] = out.reshape(
            self.get_normal_output_shape(ind=0)
        )

    # Executes the attention operator in C++ mode simulation
    def _execute_node_cppsim(self, context, graph):  # noqa: graph unused
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")

        # By convention, inputs 0, 1 and 2 correspond to named inputs q, k and v

        # Read the input from the execution context and reshape to match the
        # expected folding
        q = context[node.input[0]].reshape(self.get_folded_input_shape(ind=0))
        # Save the folded inputs to file to be used by simulation
        np.save(os.path.join(code_gen_dir, f"q.npy"), q)

        # Read the input from the execution context and reshape to match the
        # expected folding
        k = context[node.input[1]].reshape(self.get_folded_input_shape(ind=1))
        # Save the folded inputs to file to be used by simulation
        np.save(os.path.join(code_gen_dir, f"k.npy"), k)

        # Read the input from the execution context and reshape to match the
        # expected folding
        v = context[node.input[2]].reshape(self.get_folded_input_shape(ind=2))
        # Save the folded inputs to file to be used by simulation
        np.save(os.path.join(code_gen_dir, f"v.npy"), v)

        # Optionally, the mask may be provided as an input as well
        if self.get_nodeattr("mask_mode") == "input":
            # Read the input from the execution context and reshape to match the
            # expected folding
            m = context[node.input[3]].reshape(
                self.get_folded_input_shape(ind=3)
            )
            # Save the folded inputs to file to be used by simulation
            np.save(os.path.join(code_gen_dir, f"m.npy"), m)

        # Execute the precompiled model
        super().exec_precompiled_singlenode_model()

        # Load the output numpy file generated by the C++ simulation
        out = np.load(os.path.join(code_gen_dir, f"out.npy"))
        # Reshape the folded output and insert into the execution context
        context[self.onnx_node.output[0]] = out.reshape(
            self.get_normal_output_shape(ind=0)
        )

    # Executes the attention operator in RTL mode simulation
    def _execute_node_rtlsim(self, context, graph):  # noqa: graph unused
        # TODO: Implement rtlsim mode
        # Note: Cannot even compile this right now due to missing float ips
        raise NotImplementedError(
            "exec_mode rtlsim is not implemented yet!"
        )

    # Executes the attention operator in simulation (either python, c++ or rtl)
    def execute_node(self, context, graph):
        # Get the configured execution mode
        mode = self.get_nodeattr("exec_mode")
        # Lookup table mapping execution modes to implementing methods
        exec_fns = {
            "python": self._execute_node_python,
            "cppsim": self._execute_node_cppsim,
            "rtlsim": self._execute_node_rtlsim,
        }
        # Select and execute the function by mode string
        exec_fns[mode](context, graph)

    # Optional node verification
    def verify_node(self):
        pass

    # Gets the datatype of input at index ind
    def get_input_datatype(self, ind=0):
        # Ordered list of names of allowed inputs
        inputs = ["QType", "KType", "VType"]

        # If the attention mask is provided as input, it has a type as well
        if self.get_nodeattr("mask_mode") == "input":
            # The mask type is an attribute itself
            inputs += ["MType"]

        # TODO: All the following types are probably never requested, they are
        #  implemented for the sake of completeness for now. If they are ever
        #  actually required, check whether the following defaults and dummies
        #  actually still make sense.

        # If there is a thresholding activation for the first matmul, it will
        # have a type as well
        if self.get_nodeattr("ActQKMatMul") == "thresholds":
            # The thresholds will always be of the accumulator type as the
            # activation maps from AccQKMatMul to OutQKMatMul
            inputs += ["AccQKMatMul"]

        # If there is a thresholding activation for the second matmul, it will
        # have a type as well
        if self.get_nodeattr("ActAVMatMul") == "thresholds":
            # The thresholds will always be of the accumulator type as the
            # activation maps from AccAVMatMul to OutAVMatMul
            inputs += ["AccAVMatMul"]

        # If there is a thresholding activation for the softmax normalization,
        # it will have a type as well
        if self.get_nodeattr("ActASoftmax") == "thresholds":
            # While there is a dummy configurable attribute describing the
            # threshold type of the softmax, these are currently always floats
            inputs += ["AccASoftmax"]

        # Look up datatype name in attributes and convert to DataType
        return DataType[self.get_nodeattr(f"{inputs[ind]}")]

    # Gets the datatype of the output (at index ind, but there is just one)
    def get_output_datatype(self, ind=0):
        # Ordered list of names of allowed outputs
        outputs = ["O"]
        # Look up datatype name in attributes and convert to DataType
        return DataType[self.get_nodeattr(f"{outputs[ind]}Type")]

    # Gets the shape of the input at index ind without folding
    def get_normal_input_shape(self, ind=0):
        # List shapes of inputs in order
        inputs_shapes = [
            # Query input sequence
            (self.get_nodeattr("QLen"), self.get_nodeattr("QKDim")),
            # Key input sequence
            (self.get_nodeattr("KVLen"), self.get_nodeattr("QKDim")),
            # Value input sequence
            (self.get_nodeattr("KVLen"), self.get_nodeattr("VDim")),
        ]

        # If the attention mask is provided as input, it has a shape as well
        if self.get_nodeattr("mask_mode") == "input":
            # Mask shape is inferred from query and key sequence lengths
            inputs_shapes += [
                (self.get_nodeattr("QLen"), self.get_nodeattr("KVLen"))
            ]

        # TODO: All the following shapes are probably never requested, they are
        #  implemented for the sake of completeness for now. If they are ever
        #  actually required, remember to insert meaningful shapes.

        # If there is a thresholding activation for the first matmul, these will
        # be the next input index after the (optional) mask
        if self.get_nodeattr("ActQKMatMul") == "thresholds":
            # TODO: This is just a dummy shape
            inputs_shapes += [(0, 0)]

        # If there is a thresholding activation for the second matmul, these
        # will be the next input index after the (optional) first thresholds
        if self.get_nodeattr("ActAVMatMul") == "thresholds":
            # TODO: This is just a dummy shape
            inputs_shapes += [(0, 0)]

        # If there is a thresholding activation for the softmax normalization,
        # these will be the next (and last) input index after the (optional)
        # second thresholds
        if self.get_nodeattr("ActASoftmax") == "thresholds":
            # TODO: This is just a dummy shape
            inputs_shapes += [(0, 0)]

        # Get the shape by indexing into the ordered list of all inputs
        return inputs_shapes[ind]

    # Gets the shape of the output at index ind (there is just one) without
    # folding
    def get_normal_output_shape(self, ind=0):  # noqa, there is just one output
        # The output shape is inferred from the length of the query sequence and
        # the embedding dimension of the values
        return tuple((self.get_nodeattr("QLen"), self.get_nodeattr("VDim")))

    # Gets the shape of the attention weights at index ind (there is just one)
    # without folding
    def get_normal_attention_shape(self, ind=0):  # noqa, there is just one
        # The attention weights have shape covering both sequence dimensions
        return tuple((self.get_nodeattr("QLen"), self.get_nodeattr("KVLen")))

    # Gets the shape of the input at index ind with folding
    def get_folded_input_shape(self, ind=0):
        # Get the unfolded size of the input
        ilen, idim = self.get_normal_input_shape(ind)
        # Get the folding configuration specifying the amount of parallelism
        embfold, seqfold = self.folds

        # Queries, keys and values are all folded similarly along the embedding
        # dimension
        if ind in (0, 1, 2):
            # Note: Embedding dimension is always assumed to be the second
            # dimension, any transpose is handled implicitly by the operator
            return ilen, embfold, idim // embfold

        # If the mask is provided as input, it is folded along the second
        # sequence dimension
        if ind == 3 and self.get_nodeattr("mask_mode") == "input":
            # Note: Both dimensions are sequence dimension, the second
            # corresponds to the KVLen
            return ilen, seqfold, idim // seqfold

        # If this point is reached, probably something went wrong
        # TODO: Requesting the folded shape of thresholds will reach here. Can
        #  this actually happen? Probably it is indeed an error, there should be
        #  no reason to ask for the shape of the thresholds, just ask for the
        #  initializer and get its shape? Folding of the thresholds behaves
        #  differently and would require to actually keep track of mapping
        #  indices to optional inputs to correctly associate the folding
        #  dimensions.
        raise Exception(f"Requested shape of invalid input index {ind}")

    # Gets the shape of the output at index ind (there is just one) with folding
    def get_folded_output_shape(self, ind=0):  # noqa, there is just one output
        # Get the unfolded size of the output
        olen, odim = self.get_normal_output_shape(ind)
        # Get the folding configuration specifying the amount of parallelism
        embfold, seqfold = self.folds
        # The output is always folded along the embedding dimension, which is
        # assumed to be the second dimension
        return olen, embfold, odim // embfold

    # Gets the shape of the attention weights at index ind (there is just one)
    # with folding
    def get_folded_attention_shape(self, ind=0):  # noqa, there is just one
        # Get the unfolded size of the attention weights
        alen, adim = self.get_normal_attention_shape(ind)
        # Get the folding configuration specifying the amount of parallelism
        embfold, seqfold = self.folds
        # The attention weights are always folded along the sequence dimension,
        # which is assumed to be the second dimension
        return alen, seqfold, adim // seqfold

    # Widths of the input data stream of the input at index ind
    def get_instream_width(self, ind=0):
        # Get the number of bits used to represent the input
        i_bits = self.get_input_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the
        # folded input
        _, _, elems = self.get_folded_input_shape(ind)
        # Width of a stream receiving input elements in parallel
        return elems * i_bits

    # Widths of the output data stream of the output at index ind
    def get_outstream_width(self, ind=0):
        # Get the number of bits used to represent the output
        o_bits = self.get_output_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the
        # folded output
        _, _, elems = self.get_folded_output_shape(ind)
        # Width of a stream producing output elements in parallel
        return elems * o_bits

    # Maximum width of any ap_int used in this operator
    def get_ap_int_max_w(self):
        # Find the widths of the widest input
        i_bits_max = max((self.get_instream_width(ind) for ind in range(3)))
        # Find the widths of the widest output
        o_bits_max = max((self.get_outstream_width(ind) for ind in range(1)))
        # Assume no bits to represent the mask, if there is no mask
        m_bits = 0
        # A mask received as input or produced as causal on the fly has a
        # bit-width as well
        if self.get_nodeattr("mask_mode") in {"input", "causal"}:
            # Parallelism is the number of elements in the last dimension of the
            # folded mask input
            _, _, elems = self.get_folded_input_shape(ind=3)
            # Get width of the mask datatype
            m_bits = elems * DataType[self.get_nodeattr("MType")].bitwidth()

        # Elements per folded key input (second input)
        _, _, i_elems = self.get_folded_input_shape(ind=1)
        # Elements per folded value input (third input), same as the number of
        # output elements
        _, _, o_elems = self.get_folded_input_shape(ind=2)

        # Parallelism is the number of elements in the last dimension of the
        # folded attention weights
        _, _, s_elems = self.get_folded_attention_shape()
        # Number of bits used for the attention weights stream
        a_bits = s_elems * DataType[self.get_nodeattr("AType")].bitwidth()

        # Maximum bits per tile of the key and value matrix streams
        tile_bits_max = max([
            i_elems * s_elems * DataType[self.get_nodeattr("KType")].bitwidth(),
            o_elems * s_elems * DataType[self.get_nodeattr("VType")].bitwidth(),
        ])
        # Maximum bits per matmul accumulators
        acc_bits_max = max([
            # These are not streamed, thus single element width is counted
            DataType[self.get_nodeattr("AccQKMatMul")].bitwidth(),
            DataType[self.get_nodeattr("AccAVMatMul")].bitwidth(),
        ])
        # Maximum bits per matmul outputs
        out_bits_max = max([
            # These are the stream widths, which are always >= than individual
            # elements
            s_elems * DataType[self.get_nodeattr("OutQKMatMul")].bitwidth(),
            o_elems * DataType[self.get_nodeattr("OutAVMatMul")].bitwidth(),
        ])
        # Aggregate the maximum bit width in both matmul operators over all
        # inputs, intermediates and outputs
        matmul_bits_max = max([
            tile_bits_max, acc_bits_max, out_bits_max
        ])

        # Find maximum of all (maximal) bit-widths
        return max([i_bits_max, o_bits_max, m_bits, a_bits, matmul_bits_max])

    # Minimize the accumulator bit width
    def minimize_accumulator_width(self, model):  # noqa: model is unused
        # Get the query, key, value and attention weights type
        QType = DataType[self.get_nodeattr("QType")]  # noqa
        KType = DataType[self.get_nodeattr("KType")]  # noqa
        VType = DataType[self.get_nodeattr("VType")]  # noqa
        AType = DataType[self.get_nodeattr("AType")]  # noqa
        # Minimal and maximal possible results of query-key multiplication
        qk_min = self.get_nodeattr("QKDim") * QType.min() * KType.min()
        qk_max = self.get_nodeattr("QKDim") * QType.max() * KType.max()
        # Minimal and maximal possible results of attention-value multiplication
        av_min = self.get_nodeattr("KVLen") * AType.min() * VType.min()
        av_max = self.get_nodeattr("KVLen") * AType.max() * VType.max()
        # Update the accumulator types to fit the min-max range
        #   TODO: Is this correct?
        _qk_max = max(-qk_min, 1 + qk_max)
        acc_bit_width = np.log2(_qk_max) + 1
        acc_bit_width = int(np.ceil(acc_bit_width))
        self.set_nodeattr("AccQKMatMul", f"UINT{acc_bit_width}")
        _av_max = max(-av_min, 1 + av_max)
        acc_bit_width = np.log2(_av_max) + 1
        acc_bit_width = int(np.ceil(acc_bit_width))
        self.set_nodeattr("AccAVMatMul", f"UINT{acc_bit_width}")

    # Gets the number of expected output values, i.e. how many times read()
    # could/should be called on the output stream of this operator
    def get_number_output_values(self):
        # Elements over all but the last dimension of the output folded along
        # the embedding dimension
        return np.prod(self.get_folded_output_shape()[:-1])

    # Generates list of C++ includes to be placed at the top of the generated
    # code
    def global_includes(self):
        # FINN HLSLIB activation functions: e.g. PassThroughActivation
        self.code_gen_dict["$GLOBALS$"] = ['#include "activations.hpp"']
        # Attention operator HLS code
        self.code_gen_dict["$GLOBALS$"] += ['#include "attention.hpp"']

    # Converts names of optional inputs to the node input index and from there
    # to the ONNX node input name if the input is present.
    #   Note: This mapping is required as the ONNX graph/node may provide
    #   different names (in particular automatically generated unique names) and
    #   some of these are optional inputs.
    def get_input_name_by_name(self, name):
        # Ordered names of the (optional) threshold inputs
        thresholds = [
            "thresholds_qk_matmul",
            "thresholds_av_matmul",
            "thresholds_a_softmax"
        ]

        # Ordered names of primary query, key, value inputs and optional mask
        # and threshold inputs.
        inputs = ["Q", "K", "V", "M", *thresholds]

        # Specify for each input whether it is present or not
        inputs_present = [
            # Note: Primary inputs are always present, the mask is present in
            # input mask mode
            True, True, True, self.get_nodeattr("mask_mode") == "input",
        ]

        # Thresholds are present if the activation function is set to
        # thresholds
        inputs_present.extend([
            self.get_nodeattr("ActQKMatMul") == "thresholds",
            self.get_nodeattr("ActAVMatMul") == "thresholds",
            self.get_nodeattr("ActASoftmax") == "thresholds"
        ])

        # Filter the ordered list of input names for those which are actually
        # present
        inputs = [x for x, present in zip(inputs, inputs_present) if present]

        # Find the position of the requested input name and look up the
        # corresponding input name of the ONNX node
        return self.onnx_node.input[inputs.index(name)]

    # Generates C++ parameters file, i.e. activation function thresholds
    def generate_params(self, model: ModelWrapper, path):
        # The code generation directory is specified as an argument, so this
        # will work for both RTL and C++ simulation
        code_gen_dir = path

        # Note: The attention operator itself has no weights to be generated as
        # a parameter file

        # Start all three activations defaulting to pass-through of the
        # accumulator type.
        #   Note: This might allow type-casts to the output types if they are
        #   not the same as the accumulators.
        act_qk_matmul = "PassThroughActivation<AccQKMatMul>"
        act_av_matmul = "PassThroughActivation<AccAVMatMul>"
        act_a_softmax = "PassThroughActivation<float>"

        # Start all thresholds defaulting to empty default initializer braces
        thresholds_qk_matmul = "{}"
        thresholds_av_matmul = "{}"
        thresholds_a_softmax = "{}"

        # Prepares a threshold tensor as C++ string for code generation
        def prepare_thresholds(ts, length, fold, dtype):
            # Number of thresholds is given as the last dimension of the
            # threshold tensor, first dimension is covering all output elements
            num = ts.shape[-1]  # noqa
            # Partition the thresholds along the length into folds of parallel
            # elements
            ts = interleave_matrix_outer_dim_from_partitions(ts, length // fold)
            # Reshape folded thresholds adding an outer dimension
            # TODO: Why? MVAU does this, just copied the behavior. This is
            #  probably to generate the outer C++ initializer braces {} for
            #  object construction. Isn't it weird to rely on an artificial
            #  dimension just to have the code generator produce the correct
            #  string?
            ts = ts.reshape(1, length // fold, fold, num)
            # Format the thresholds as C++ array code
            # Note: no packing, no variable name/type declaration
            return numpy_to_hls_code(ts, dtype, "_", False, True), num

        # Get shape and folding configuration. None of the activations fold
        # along the query-key embedding dimension or the query sequence length
        (_, _, vdim, kvlen), (embfold, seqfold) = self.shapes, self.folds

        # Query-key matmul can have an optional activation function set to
        # thresholding activations via node attribute
        if self.get_nodeattr("ActQKMatMul") == "thresholds":
            # In this case there will be a thresholds parameter initializer
            thresholds = model.get_initializer(
                self.get_input_name_by_name("thresholds_qk_matmul")
            )
            # Get the datatype of the thresholds
            thresholds_dtype = DataType[self.get_nodeattr("AccQKMatMul")]
            # Activation value, i.e., bias applied after thresholding activation
            bias = self.get_nodeattr("BiasActQKMatMul")
            # No support for floating-point bias
            assert int(bias) == bias, "BiasActQKMatMul must be integer"
            # Convert the bias to integer representation, so it can be used as a
            # template argument
            bias = int(bias)
            # Format the thresholds as C++ array code: QK matmul outputs fold
            # along the key-value sequence length dimension
            thresholds_qk_matmul, num = prepare_thresholds(
                thresholds, kvlen, seqfold, thresholds_dtype
            )
            # Replace default pass-through activation by thresholding activation
            #   Note: Relies on type and shape definitions generated by the
            #   "defines" method
            act_qk_matmul = "\n".join([
                f"ThresholdsActivation<",
                f" SeqFold,"
                f" KVLen/SeqFold,"
                f" {num},"
                f" AccQKMatMul,"
                f" OutQKMatMul,"
                f" {bias}",
                f">"
            ])

        # Attention-value matmul can have an optional activation function set to
        # thresholding activations via node attribute
        if self.get_nodeattr("ActAVMatMul") == "thresholds":
            # In this case there will be a thresholds parameter initializer
            thresholds = model.get_initializer(
                self.get_input_name_by_name("thresholds_av_matmul")
            )
            # Get the datatype of the thresholds
            thresholds_dtype = DataType[self.get_nodeattr("AccAVMatMul")]
            # Activation value, i.e., bias applied after thresholding activation
            bias = self.get_nodeattr("BiasActAVMatMul")
            # No support for floating-point bias
            assert int(bias) == bias, "BiasActAVMatMul must be integer"
            # Convert the bias to integer representation, so it can be used as a
            # template argument
            bias = int(bias)
            # Format the thresholds as C++ array code: AV matmul outputs fold
            # along the value embedding dimension
            thresholds_av_matmul, num = prepare_thresholds(
                thresholds, vdim, embfold, thresholds_dtype
            )
            # Replace default pass-through activation by thresholding activation
            #   Note: Relies on type and shape definitions generated by the
            #   "defines" method
            act_av_matmul = "\n".join([
                f"ThresholdsActivation<",
                f" EmbFold,"
                f" VDim/EmbFold,"
                f" {num},"
                f" AccAVMatMul,"
                f" OutAVMatMul,"
                f" {bias}"
                f">"
            ])

        # Softmax can have an optional activation function set to thresholding
        # activations via node attribute
        if self.get_nodeattr("ActASoftmax") == "thresholds":
            # In this case there will be a thresholds parameter initializer
            thresholds = model.get_initializer(
                self.get_input_name_by_name("thresholds_a_softmax")
            )
            # Get the datatype of the thresholds
            thresholds_dtype = DataType[self.get_nodeattr("AccASoftmax")]
            # Activation value, i.e., bias applied after thresholding activation
            bias = self.get_nodeattr("BiasActASoftmax")
            # No support for floating-point bias
            assert int(bias) == bias, "BiasActASoftmax must be integer"
            # Convert the bias to integer representation, so it can be used as a
            # template argument
            bias = int(bias)
            # Format the thresholds as C++ array code: Softmax outputs fold
            # along the key-value sequence length dimension
            thresholds_a_softmax, num = prepare_thresholds(
                thresholds, kvlen, seqfold, thresholds_dtype
            )
            # Replace default pass-through activation by thresholding activation
            #   Note: Relies on type and shape definitions generated by the
            #   "defines" method
            act_a_softmax = "\n".join([
                f"ThresholdsActivation<",
                f" SeqFold,"
                f" KVLen/SeqFold,"
                f" {num},"
                f" AccASoftmax,"
                f" AType,"
                f" {bias}",
                f">"
            ])

        # Open a file to store the thresholds parameters as C++ code
        with open(f"{code_gen_dir}/params.hpp", "w") as file:
            # Write lines of C++ code separated by newlines to the file
            file.write("\n".join([
                # Scale factor preceding the softmax activation function to
                # dequantize the input to floating-point representation
                f"static const float dequant_softmax ="
                f" {self.get_nodeattr('DequantSoftmax')};",
                # Add type definition and threshold initialization of the
                # query-key matmul activation
                f"using ActQKMatMul = {act_qk_matmul};",
                f"ActQKMatMul act_qk_matmul = {thresholds_qk_matmul};",
                # Add type definition and threshold initialization of the
                # attention-value matmul activation
                f"using ActAVMatMul = {act_av_matmul};",
                f"ActAVMatMul act_av_matmul = {thresholds_av_matmul};",
                # Add type definition and threshold initialization of the
                # softmax activation
                f"using ActASoftmax = {act_a_softmax};",
                f"ActASoftmax act_a_softmax = {thresholds_a_softmax};",
                # Append a newline at the end of the file (to avoid problems
                # when including, required by C standard?)
                "\n"
            ]))

    # Generates C++ code of type alias, global constant and macro definitions
    def defines(self, var):
        # Generate shape definitions from attributes to C++ constant definitions
        def shapedefs(*names):
            # C++ qualified type to be used for shape constants
            shape = "static constexpr std::size_t"
            # Generate a C++ constant definition for each of the attributes
            # given by argument list names
            return (
                f"{shape} {name} = {self.get_nodeattr(name)};" for name in names
            )

        # Generate datatype definitions mapping from QONNX DataType to HLS type
        def typedefs(*names):
            # Gets the HLS type string for the datatype specified by the named
            # attribute
            def hls_type(name):
                # Looks up the datatype specified for the attribute and
                # translates from QONNX to HLS type
                return DataType[self.get_nodeattr(name)].get_hls_datatype_str()

            # Generate a C++ type alias definition for each of the attributes
            # given by argument list names
            return (f"using {name} = {hls_type(name)};" for name in names)

        # Insert constants and type aliases into the dictionary
        self.code_gen_dict["$DEFINES$"] = [
            # Shape constant definitions of attention inputs (query, key and
            # value) and folding configuration
            *shapedefs(
                "QKDim",
                "QLen",
                "VDim",
                "KVLen",
                "EmbFold",
                "SeqFold"
            ),
            # Type alias definitions for all input, output and intermediate
            # datatypes
            *typedefs(
                "QType",
                "KType",
                "VType",
                "MType",
                "AType",
                "OType"
            ),
            # Type alias definitions for the matmul accumulators and output
            # datatypes
            *typedefs(
                "AccQKMatMul",
                "OutQKMatMul",
                "AccAVMatMul",
                "OutAVMatMul",
                "AccASoftmax"
            ),
            # Include the activation function type definitions and parameters
            #   Note: The typedefs in this header require the typedefs above,
            #   thus adding this to the global includes is not possible.
            f'#include "params.hpp"',
            # Type alias of the properly configured attention operator class
            f"using Attention = ScaledDotProductAttention<",
            f"    QKDim,",
            f"    QLen,",
            f"    VDim,",
            f"    KVLen,",
            f"    EmbFold,",
            f"    SeqFold,",
            f"    QType,",
            f"    KType,",
            f"    VType,",
            f"    MType,",
            f"    AType,",
            f"    OType,",  # Note: OType and last MatMul out must match
            f"    AccQKMatMul,",
            f"    OutQKMatMul,",
            f"    ActQKMatMul,",
            f"    AccAVMatMul,",
            f"    OType,",  # Note: OType and last MatMul out must match
            f"    ActAVMatMul,",
            f"    ActASoftmax",
            f">;",
            # Short type aliases of attention input and output streams
            f"using QStream = Attention::QStream;",
            f"using KStream = Attention::KStream;",
            f"using VStream = Attention::VStream;",
            f"using OStream = Attention::OStream;",
            f"using MStream = Attention::MStream;",
        ]

    # Generates C++ code for reading data from .npy (numpy format) for testing
    # in C++ simulation
    def read_npy_data(self):
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")

        # Generate function calls for reading the input files into the input
        # streams
        self.code_gen_dict["$READNPYDATA$"] = [
            # Deduce the datatype of elements packed into the query input stream
            #   TODO: Maybe these type-deductions can be removed by changing the
            #    order of the template arguments of the npy2apintstream, such
            #    that type-deduction is handled there?
            f'using QPacked = decltype(QStream{{}}.read());',
            # Generate function call reading from file into the input stream
            #   Note: Inputs are always represented as numpy floats
            f'npy2apintstream<QPacked, QType, QType::width, float>(',
            f'  "{code_gen_dir}/q.npy", q_{self.hls_sname()}, false',
            ');',

            # Deduce the datatype of elements packed into the key input stream
            f'using KPacked = decltype(KStream{{}}.read());',
            # Generate function call reading from file into the input stream
            #   Note: Inputs are always represented as numpy floats
            f'npy2apintstream<KPacked, KType, KType::width, float>(',
            f'  "{code_gen_dir}/k.npy", k_{self.hls_sname()}, false',
            ');',

            # Deduce the datatype of elements packed into the value input stream
            f'using VPacked = decltype(VStream{{}}.read());',
            # Generate function call reading from file into the input stream
            #   Note: Inputs are always represented as numpy floats
            f'npy2apintstream<VPacked, VType, VType::width, float>(',
            f'  "{code_gen_dir}/v.npy", v_{self.hls_sname()}, false',
            ');',
        ]

        # If the mask is provided as an input, it needs to be read as well
        if self.get_nodeattr("mask_mode") == "input":
            # Generate function call for reading the mask file into the input
            # stream
            self.code_gen_dict["$READNPYDATA$"] += [
                # Deduce the datatype of elements packed into the mask input
                # stream
                f'using MPacked = decltype(MStream{{}}.read());',
                # Generate function call reading from file into the input stream
                #   Note: Inputs are always represented as numpy floats
                f'npy2apintstream<MPacked, MType, MType::width, float>(',
                f'  "{code_gen_dir}/m.npy", m_{self.hls_sname()}, false',
                ');',
            ]

    # Generates C++ code for declaring all streams involved in C++ simulation
    # for testing
    def strm_decl(self):
        # Declare input (query, key, value) and output streams
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            # Note: Assumes stream type aliases to be set in defines
            f"QStream q_{self.hls_sname()};",
            f"KStream k_{self.hls_sname()};",
            f"VStream v_{self.hls_sname()};",
            f"OStream out_{self.hls_sname()};"
        ]
        # If the mask is provided as an input, it needs a stream declaration as
        # well
        if self.get_nodeattr("mask_mode") == "input":
            # Append the mask stream to the declaration list
            self.code_gen_dict["$STREAMDECLARATIONS$"] += [
                # Note: Assumes stream type aliases to be set in defines
                f"MStream m_{self.hls_sname()};",
            ]

    # Generates C++ code for calling the computation part of the operator
    def docompute(self):
        # Write the body of the attention top-level function
        self.code_gen_dict["$DOCOMPUTE$"] = [
            # Instantiate the attention operator and connect to the generated
            # threshold parameters
            # Note: Assumes "Attention" to be aliased appropriate configuration
            #   in defines with.
            # Note: Assumes parameters to be generated in 'generate_params' and
            #   made available via include/defines before.
            f"Attention attention {{",
            f"    act_qk_matmul, act_av_matmul, act_a_softmax, dequant_softmax",
            f"}};",
            # Connect the attention operator to the input and output streams
            f"attention("
            f"q_{self.hls_sname()}, "
            f"k_{self.hls_sname()}, "
            f"v_{self.hls_sname()}, "
            f"out_{self.hls_sname()}"
            f");",
        ]

    # Generates C++ code for reading the output stream and converting back to
    # numpy format for testing in C** simulation
    def dataoutstrm(self):
        # Output data will be stored in numpy files in the code generation
        # dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        # Get the expected shape of the folded output array formatted as a C++
        # vector initializer
        # Note: Valid formatting relies on correct placement of curly braces
        # and line breaks: Open/close all three braces on the same line of code
        # to avoid '\n' to be inserted into the string
        shape = f"""{{{
        ','.join((str(i) for i in self.get_folded_output_shape()))
        }}}"""
        # Generate function call for reading from the output stream into the
        # output file
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            # Deduce the datatype of elements packed into the output stream
            f'using OPacked = decltype(OStream{{}}.read());',
            # Generate function call reading from stream into the output file
            #   Note: Outputs are always represented as numpy floats
            f'apintstream2npy<OPacked, OType, OType::width, float>(',
            f'out_{self.hls_sname()}, {shape}, "{code_gen_dir}/out.npy", false',
            ');',
        ]

    # Generates C++ code for saving the output of C++ simulation to a file in
    # numpy format
    def save_as_npy(self):
        # Note: This seems to be empty in ALL HLSCustomOps. Probably it was used
        # for something before, which is now integrated into dataoutstrm()?
        self.code_gen_dict["$SAVEASCNPY$"] = []

    # Generates essentially the head of the C++ function from which the IP block
    # will be generated during ipgen, i.e. actual synthesis
    def blackboxfunction(self):
        # Insert function head describing the top level interface of the
        # attention operator
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            # Note: Assumes stream type aliases to be set in defines
            f"void {self.onnx_node.name} (",
            f"  QStream &q_{self.hls_sname()},"
            f"  KStream &k_{self.hls_sname()},"
            f"  VStream &v_{self.hls_sname()},"
            f"  OStream &out_{self.hls_sname()}",
            f")",
        ]

    # Generates C++ pragmas to be inserted into the main function of the C++
    # simulation and the ipgen-blackboxfunction as well
    def pragmas(self):
        # Add HLS interface directives specifying how to create RTL ports for
        # the top-level function arguments
        self.code_gen_dict["$PRAGMAS$"] = [
            # Connect the query input stream with an axi stream interface
            f"#pragma HLS INTERFACE axis port=q_{self.hls_sname()}",
            # Connect the key input stream with an axi stream interface
            f"#pragma HLS INTERFACE axis port=k_{self.hls_sname()}",
            # Connect the value input stream with an axi stream interface
            f"#pragma HLS INTERFACE axis port=v_{self.hls_sname()}",
            # Connect the output stream with an axi stream interface
            f"#pragma HLS INTERFACE axis port=out_{self.hls_sname()}",
        ]
        # No block-level I/O protocol for the function return value
        self.code_gen_dict["$PRAGMAS$"].append(
            f"#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    # Returns the names of input and output interfaces grouped by protocol
    def get_verilog_top_module_intf_names(self):
        # Start collecting interface names in a dictionary starting with clock
        # and reset
        intf_names = {"clk": ["ap_clk"], "rst": ["ap_rst_n"]}  # noqa
        # AXI stream input interfaces
        intf_names["s_axis"] = [
            (f"q_{self.hls_sname()}", self.get_instream_width_padded(ind=0)),
            (f"k_{self.hls_sname()}", self.get_instream_width_padded(ind=1)),
            (f"v_{self.hls_sname()}", self.get_instream_width_padded(ind=2))
        ]
        # AXI stream output interfaces
        intf_names["m_axis"] = [
            (f"out_{self.hls_sname()}", self.get_outstream_width_padded(ind=0))
        ]
        # No AXI-MM, AXI-Lite or protocol-less interfaces
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        # Return the interface name dictionary
        return intf_names
