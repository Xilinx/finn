from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl
import numpy as np
import os
from pyverilator.util.axi_utils import reset_rtlsim, toggle_clk
from qonnx.core.datatype import DataType
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import get_dsp_block, get_rtlsim_trace_depth, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

class DynMVAU_rtl(MVAU_rtl):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):

        my_attrs = {
            "N_VECTORS": ("i", True, 0) # Height of Matrix A
        }

        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def generate_params(self, model, path):
        # Dynamic MVAU does not have weight parameters
        pass

    def generate_hdl(self, model, fpgapart, clk):
        # Generate params as part of IP preparation
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        template_path, code_gen_dict = self.prepare_codegen_default(fpgapart, clk)

        code_gen_dict["$NARROW_WEIGHTS$"] = str(0)
        # add general parameters to dictionary
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [self.get_verilog_top_module_name()]
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # apply code generation to template
        with open(template_path, "r") as f:
            template_wrapper = f.read()
        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            print(f"Replacing {key} with {code_gen_line}")
            template_wrapper = template_wrapper.replace(key, code_gen_line)
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"),
            "w",
        ) as f:
            f.write(template_wrapper.replace("$FORCE_BEHAVIORAL$", str(0)))
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper_sim.v"),
            "w",
        ) as f:
            f.write(template_wrapper.replace("$FORCE_BEHAVIORAL$", str(1)))

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)


    def prepare_codegen_default(self, fpgapart, clk):
        template_path = os.environ["FINN_ROOT"] + "/finn-rtllib/mvu/mvu_dyn_axi_wrapper.v"

        dsp_block = get_dsp_block(fpgapart)
        code_gen_dict = {}
        code_gen_dict["$IS_MVU$"] = [str(1)]
        code_gen_dict["$COMPUTE_CORE$"] = [self._resolve_impl_style(dsp_block)]
        code_gen_dict["$MW$"] = [str(self.get_nodeattr("MW"))]
        code_gen_dict["$MH$"] = [str(self.get_nodeattr("MH"))]
        code_gen_dict["$PE$"] = [str(self.get_nodeattr("PE"))]
        code_gen_dict["$SIMD$"] = [str(self.get_nodeattr("SIMD"))]
        code_gen_dict["$ACTIVATION_WIDTH$"] = [str(self.get_input_datatype(0).bitwidth())]
        code_gen_dict["$WEIGHT_WIDTH$"] = [str(self.get_input_datatype(1).bitwidth())]
        code_gen_dict["$ACCU_WIDTH$"] = [str(self.get_output_datatype().bitwidth())]
        code_gen_dict["$N_VECTORS$"] = [str(self.get_output_datatype().bitwidth())]
        code_gen_dict["$SIGNED_ACTIVATIONS$"] = (
            [str(1)] if (self.get_input_datatype(0).min() < 0) else [str(0)]
        )
        code_gen_dict["$SEGMENTLEN$"] = [str(self._resolve_segment_len(clk))]

        return template_path, code_gen_dict
