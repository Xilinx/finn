from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import roundup_to_integer_multiple


class AssignMemoryOffset(Transformation):
    def apply(self, model):
        self._offset = 0
        self._walk(model)
        return model, False

    def _walk(self, model):
        for node in model.graph.node:
            if node.op_type == "FINNLoop":
                loop_inst = getCustomOp(node)
                body_model = loop_inst.get_nodeattr("body")
                self._walk(body_model)
                loop_inst.set_nodeattr("body", body_model.graph)
                loop_inst.set_nodeattr("address_offset", self._offset)
                self._offset += loop_inst.intermediate_frame_bytes()
                self._offset = roundup_to_integer_multiple(self._offset, 32)
            else:
                inst = getCustomOp(node)
                mlo_max_iter = inst.get_nodeattr("mlo_max_iter")
                if not mlo_max_iter or not node.op_type == "MVAU_rtl":
                    continue
                inst.set_nodeattr("address_offset", self._offset)
                self._offset += mlo_max_iter * inst.wmat_size_bytes()
                self._offset = roundup_to_integer_multiple(self._offset, 32)
