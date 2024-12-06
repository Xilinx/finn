from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl
import numpy as np
import os
from pyverilator.util.axi_utils import reset_rtlsim, toggle_clk
from qonnx.core.datatype import DataType
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import get_dsp_block, get_rtlsim_trace_depth, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None
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
        code_gen_dict["$N_VECTORS$"] = [str(self.get_nodeattr("N_VECTORS"))]
        code_gen_dict["$SIGNED_ACTIVATIONS$"] = (
            [str(1)] if (self.get_input_datatype(0).min() < 0) else [str(0)]
        )
        code_gen_dict["$SEGMENTLEN$"] = [str(self._resolve_segment_len(clk))]

        return template_path, code_gen_dict


    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""

        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # Path to (System-)Verilog files used by top-module & path to top-module
        verilog_paths = [code_gen_dir, os.environ["FINN_ROOT"] + "/finn-rtllib/mvu"]
        verilog_files = [self.get_nodeattr("gen_top_module") + "_wrapper_sim.v"]

        # build the Verilator emu library
        sim = PyVerilator.build(
            verilog_files,
            build_dir=make_build_dir("pyverilator_" + self.onnx_node.name + "_"),
            verilog_path=verilog_paths,
            trace_depth=get_rtlsim_trace_depth(),
            top_module_name=self.get_verilog_top_module_name(),
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", sim.lib._name)

        return sim

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        mem_mode = self.get_nodeattr("mem_mode")
        node = self.onnx_node

        if mode == "cppsim":
            MVAU.execute_node(self, context, graph)
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            # create a npy file fore each input of the node (in_ind is input index)
            in_ind = 0
            for inputs in node.input:
                # it is assumed that the first input of the node is the data input
                # the second input are the weights
                if in_ind < 2:
                    assert (
                        str(context[inputs].dtype) == "float32"
                    ), """Input datatype is
                    not float32 as expected."""
                    expected_inp_shape = self.get_folded_input_shape(in_ind)

                    reshaped_input = context[inputs].reshape(expected_inp_shape)
                    export_idt = self.get_input_datatype()
                    # make copy before saving the array
                    reshaped_input = reshaped_input.copy()
                    np.save(
                        os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                        reshaped_input,
                    )
                else:
                    raise Exception("Unexpected input found for MatrixVectorActivation_rtl")
                in_ind += 1

            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            inp_0 = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
            reset_rtlsim(sim)
            toggle_clk(sim)
            if mem_mode in ["external", "internal_decoupled"]:
                wnbits = self.get_weightstream_width()
                export_wdt = self.get_weight_datatype()
                inp_1 = npy_to_rtlsim_input(
                    "{}/input_1.npy".format(code_gen_dir), export_wdt, wnbits
                )
                num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
                io_dict = {
                    "inputs": {"in0": inp_0, "in1": inp_1 * num_w_reps},
                    "outputs": {"out": []},
                }
                self.rtlsim_multi_io(sim, io_dict)
                output = io_dict["outputs"]["out"]
            else:
                output = self.rtlsim(sim, inp)
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(output, out_npy_path, odt, out_shape, packed_bits, target_bits)
            # load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        mem_mode = self.get_nodeattr("mem_mode")
        sname = self.hls_sname()
        if mem_mode == "external":
            # find the weights_V interface and rename it to in1_V
            for i, (name, width) in enumerate(intf_names["s_axis"]):
                if name == "weights_V":
                    intf_names["s_axis"][i] = ("in1_" + sname, self.get_weightstream_width_padded())
            # intf_names["s_axis"].append(("in1_" + sname, self.get_weightstream_width_padded()))
        if mem_mode == "internal_decoupled":
            # only expose axilite interface if attribute is set
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            if runtime_writable:
                intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def get_folded_input_shape(self, ind=0):
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")
        sf = mw // simd
        nf = mh // pe
        if ind == 0:
            # calculate shape of input 0
            vecs = list(self.get_nodeattr("numInputVectors"))
            folded_input_shape = tuple(vecs + [sf, simd])
        elif ind == 1:
            # calculate shape of input 1
            vecs = [self.get_nodeattr("MW")]
            folded_input_shape = tuple(vecs + [nf, pe])
        else:
            raise Exception("Undefined input shape for requested input")

        return folded_input_shape

    def instantiate_ip(self, cmd):
        # instantiate the RTL IP
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/")
        sourcefiles = [
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"),
            rtllib_dir + "mv_matrix_load.sv",
            rtllib_dir + "mv_matrix_load_wide.sv",
            rtllib_dir + "mvu_4sx4u.sv",
            rtllib_dir + "mvu_8sx8u_dsp48.sv",
            rtllib_dir + "mvu_dyn_axi.sv",
            # rtllib_dir + "mvu_dyn_axi_wrapper.v",
            rtllib_dir + "mvu_vvu_8sx9_dsp58.sv",
            rtllib_dir + "mvu_vvu_axi.sv",
            # rtllib_dir + "mvu_vvu_axi_wrapper.v",
            rtllib_dir + "ram_p_c.sv",
            rtllib_dir + "replay_buffer.sv"
        ]
        for f in sourcefiles:
            cmd.append("add_files -norecurse %s" % (f))
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (
                    self.get_nodeattr("gen_top_module"),
                    self.onnx_node.name,
                    self.onnx_node.name,
                )
            )
        else:
            cmd.append(
                "create_bd_cell -type hier -reference %s %s"
                % (
                    self.get_nodeattr("gen_top_module"),
                    self.onnx_node.name,
                )
            )
