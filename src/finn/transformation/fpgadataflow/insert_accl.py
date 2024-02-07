from onnx import TensorProto
from onnx import helper as oh
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph


class InsertACCL(Transformation):
    def insert_at(self, model, world_size, tensor_name, producer, consumer):
        if producer.op_type == "ACCLOut":
            assert consumer.op_type == "ACCLIn", "Expect ACCLOut to comer after in"
            return False

        producer_inst = getCustomOp(producer)
        consumer_inst = getCustomOp(consumer)

        producer_rank = producer_inst.get_nodeattr("device_id")
        consumer_rank = consumer_inst.get_nodeattr("device_id")

        # Nodes are on same device, no need to insert accl nodes
        if producer_rank == consumer_rank:
            return False

        tensor_shape = model.get_tensor_shape(tensor_name)
        tensor_dtype = model.get_tensor_datatype(tensor_name)

        producer_out = oh.make_tensor_value_info(
            model.make_new_valueinfo_name(), TensorProto.FLOAT, tensor_shape
        )

        model.graph.value_info.append(producer_out)
        model.set_tensor_datatype(producer_out.name, tensor_dtype)

        consumer_in = oh.make_tensor_value_info(
            model.make_new_valueinfo_name(), TensorProto.FLOAT, tensor_shape
        )

        model.graph.value_info.append(consumer_in)
        model.set_tensor_datatype(consumer_in.name, tensor_dtype)

        producer_shape = producer_inst.get_folded_output_shape()

        for idx, out in enumerate(producer.output):
            if out == tensor_name:
                producer.output[idx] = producer_out.name

        cmd_to_cclo = oh.make_tensor_value_info(
            f"cmd_to_cclo_{model.make_new_valueinfo_name()}", TensorProto.UINT32, [15, 1]
        )
        model.graph.output.append(cmd_to_cclo)

        sts_from_cclo = oh.make_tensor_value_info(
            f"sts_from_cclo_{model.make_new_valueinfo_name()}", TensorProto.UINT32, [1, 1]
        )
        model.graph.input.append(sts_from_cclo)

        accl_out = oh.make_node(
            "ACCLOut",
            [producer_out.name, sts_from_cclo.name],
            [tensor_name, cmd_to_cclo.name],
            numInputVectors=producer_shape[:-1],
            NumChannels=producer_shape[-1],
            dataType=str(tensor_dtype),
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            device_id=producer_rank,
            worldSize=world_size,
            otherRank=consumer_rank,
        )

        # As we are sorting the graph afterwards it should be fine to insert this at
        # beginning
        model.graph.node.insert(0, accl_out)

        consumer_shape = consumer_inst.get_folded_input_shape()

        accl_in = oh.make_node(
            "ACCLIn",
            [tensor_name],
            [consumer_in.name],
            numInputVectors=consumer_shape[:-1],
            NumChannels=consumer_shape[-1],
            dataType=str(tensor_dtype),
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            device_id=consumer_rank,
            worldSize=world_size,
            otherRank=producer_rank,
        )

        model.graph.node.insert(0, accl_in)

        for idx, inp in enumerate(consumer.input):
            if inp == tensor_name:
                consumer.input[idx] = consumer_in.name

        return True

    def apply(self, model):
        world_size = 1
        for node in model.graph.node:
            node_inst = getCustomOp(node)
            world_size = max(world_size, node_inst.get_nodeattr("device_id") + 1)

        potential_comm_pairs = []

        for producer in model.graph.node:
            for tensor_name in producer.output:
                consumer = model.find_consumer(tensor_name)
                if consumer is None:
                    continue
                potential_comm_pairs.append((tensor_name, producer, consumer))

        modified = False

        for tensor_name, producer, consumer in potential_comm_pairs:
            modified |= self.insert_at(model, world_size, tensor_name, producer, consumer)

        if modified:
            model = model.transform(SortGraph())

        return (model, modified)
