from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor, roundup_to_integer_multiple
import finn.util.data_packing as dpk
from finn.util.data_packing import (
    hexstring2npbytearray,
    pack_innermost_dim_as_hex_string,
)
from typing import Dict
import numpy as np


# TODO: License?

def to_external_tensor(init, w_dtype):
    """Return an appropriately formatted and packed numpy byte array for given
    external parameter tensor."""

    weight_width = init.shape[1] * w_dtype.bitwidth()
    weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
    hex_init = pack_innermost_dim_as_hex_string(init, w_dtype, weight_width_padded, prefix="0x")
    ext_weight = np.array([], dtype=np.uint8)
    for line in hex_init:
        array_line = [x for x in reversed(hexstring2npbytearray(line, remove_prefix="0x"))]
        ext_weight = np.append(ext_weight, array_line)

    return ext_weight


def get_driver_shapes(model: ModelWrapper) -> Dict:
    idt = []
    idma_names = []
    ishape_normal = []
    ishape_folded = []
    ishape_packed = []
    for idma_ind, graph_in in enumerate(model.graph.input):
        i_tensor_name = graph_in.name
        # get inp tensor properties
        i_tensor_dt = model.get_tensor_datatype(i_tensor_name)
        i_tensor_shape_normal = tuple(model.get_tensor_shape(i_tensor_name))
        # go down into dataflow partition to get folded shape info etc
        # TODO consider setting these as attributes during dataflow partitioning
        i_consumer = model.find_consumer(i_tensor_name)
        assert (
            i_consumer.op_type == "StreamingDataflowPartition"
        ), """
            Ensure CreateDataflowPartition called before driver creation."""
        first_df_model = ModelWrapper(getCustomOp(i_consumer).get_nodeattr("model"))
        assert (
            first_df_model.graph.node[0].op_type == "IODMA_hls"
        ), "First partition must hold input IODMA"
        successors = model.find_direct_successors(i_consumer)
        successor_input_num = list(successors[0].input).index(i_consumer.output[0])
        successor_sdp = getCustomOp(successors[0])
        successor_df_model = ModelWrapper(successor_sdp.get_nodeattr("model"))
        first_node = successor_df_model.find_consumer(
            successor_df_model.graph.input[successor_input_num].name
        )
        i_tensor_shape_folded = tuple(getCustomOp(first_node).get_folded_input_shape())
        # generate dummy folded i/o tensors and their packed versions
        i_tensor_dummy_folded = gen_finn_dt_tensor(i_tensor_dt, i_tensor_shape_folded)
        i_tensor_dummy_packed = dpk.finnpy_to_packed_bytearray(
            i_tensor_dummy_folded, i_tensor_dt
        )
        i_tensor_shape_packed = i_tensor_dummy_packed.shape
        # append all input tensor info to relevant lists
        idt.append("DataType['%s']" % i_tensor_dt.name)
        ishape_normal.append(i_tensor_shape_normal)
        ishape_folded.append(i_tensor_shape_folded)
        ishape_packed.append(i_tensor_shape_packed)
        idma_names.append(getCustomOp(i_consumer).get_nodeattr("instance_name"))

    odt = []
    odma_names = []
    oshape_normal = []
    oshape_folded = []
    oshape_packed = []
    for odma_ind, graph_out in enumerate(model.graph.output):
        o_tensor_name = graph_out.name
        # get inp tensor properties
        o_tensor_dt = model.get_tensor_datatype(o_tensor_name)
        o_tensor_shape_normal = tuple(model.get_tensor_shape(o_tensor_name))
        # go down into IODMA partition to get folded shape info etc
        # TODO consider setting these as attributes during dataflow partitioning
        o_producer = model.find_producer(o_tensor_name)
        assert (
            o_producer.op_type == "StreamingDataflowPartition"
        ), """
            Ensure CreateDataflowPartition called before driver creation."""
        df_model = ModelWrapper(getCustomOp(o_producer).get_nodeattr("model"))
        assert df_model.graph.node[-1].op_type == "IODMA_hls", "Partition must hold output IODMA"
        predecessors = model.find_direct_predecessors(o_producer)
        predecessor_output_num = list(predecessors[0].output).index(o_producer.input[0])
        predecessor_sdp = getCustomOp(predecessors[0])
        predecessor_df_model = ModelWrapper(predecessor_sdp.get_nodeattr("model"))
        last_node = predecessor_df_model.find_producer(
            predecessor_df_model.graph.output[predecessor_output_num].name
        )
        o_tensor_shape_folded = tuple(getCustomOp(last_node).get_folded_output_shape())
        o_tensor_dummy_folded = gen_finn_dt_tensor(o_tensor_dt, o_tensor_shape_folded)
        o_tensor_dummy_packed = dpk.finnpy_to_packed_bytearray(
            o_tensor_dummy_folded, o_tensor_dt
        )
        o_tensor_shape_packed = o_tensor_dummy_packed.shape
        # append all output tensor info to relevant lists
        odt.append("DataType['%s']" % o_tensor_dt.name)
        oshape_normal.append(o_tensor_shape_normal)
        oshape_folded.append(o_tensor_shape_folded)
        oshape_packed.append(o_tensor_shape_packed)
        odma_names.append(getCustomOp(o_producer).get_nodeattr("instance_name"))
    
    return {
        "idt": idt,
        "idma_names": idma_names,
        "ishape_normal": ishape_normal,
        "ishape_folded": ishape_folded,
        "ishape_packed": ishape_packed,

        "odt": odt,
        "odma_names": odma_names,
        "oshape_normal": oshape_normal,
        "oshape_folded": oshape_folded,
        "oshape_packed": oshape_packed,
    }