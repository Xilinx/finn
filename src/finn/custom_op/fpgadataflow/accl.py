import math
import numpy as np
import warnings
import time
import threading
import traceback
from collections import defaultdict
import psutil

from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp

from IPython.core.debugger import set_trace

import subprocess
import os

accl_word_size = 512

class ACCLOp(HLSCustomOp):
    barriers = defaultdict(lambda: threading.Barrier(2))

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "dataType": ("s", True, ""),
            # shape describing input vecs per execution
            "numInputVectors": ("ints", False, [1]),
            # ACCL specific attrs
            "startPort": ("i", False, 5500),
            "worldSize": ("i", True, 0),
            "otherRank": ("i", True, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        vecs = list(self.get_nodeattr("numInputVectors"))
        num_ch = self.get_nodeattr("NumChannels")
        ishape = tuple(vecs + [num_ch])
        return ishape

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def compile_singlenode_code(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")

        build_dir = code_gen_dir
        os.makedirs(build_dir, exist_ok=True)

        subprocess.run([
                "/usr/bin/cmake",
                f"{os.environ['FINN_ROOT']}/custom_hls/accl",
                f"-DCODE_GEN_DIR={code_gen_dir}",
            ],
            cwd=build_dir,
            stdout=subprocess.PIPE,
        )
        subprocess.run(["make"], cwd=build_dir)

        self.set_nodeattr("executable_path", code_gen_dir + "/node_model")

    def execute_op(self, edge_name):
        idx = ACCLOp.barriers[edge_name].wait()
        emulator = None
        try:
            if idx == 0:
                emulator_dir = f"{os.environ['FINN_ROOT']}/ACCL/test/model/emulator"
                world_size = self.get_nodeattr("worldSize")

                subprocess.run(["/usr/bin/cmake", "."],
                               cwd=emulator_dir, stdout=subprocess.PIPE)

                emulator = subprocess.Popen([
                    "python3",
                    "run.py",
                    f"-n {world_size}",
                    "--no-kernel-loopback"
                ], cwd=emulator_dir)

            executable_path = self.get_nodeattr("executable_path")
            if executable_path == "":
                raise Exception(
                    """
Found no executable for this node, did you run the codegen and
compilation transformations?
                """
                )

            p = subprocess.Popen(
                executable_path,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                encoding="utf-8"
            )

            while line := p.stdout.readline():
                print(line, end='')
                if "CCLO BFM started" in line:
                    break
            else:
                raise Exception("Process did not signal that CCLO was started")

            ACCLOp.barriers[edge_name].wait()
            p.communicate("...")
            idx = ACCLOp.barriers[edge_name].wait()
        except Exception:
            print(traceback.format_exc())
            ACCLOp.barriers[edge_name].abort()
        finally:
            if emulator is not None:
                parent_proc = psutil.Process(emulator.pid)
                for child in parent_proc.children(recursive=True):
                    child.kill()
                emulator.kill()

    def get_number_output_values(self):
        oshape = self.get_normal_output_shape()
        itype_bits = self.get_input_datatype().bitwidth()
        stream_width = self.get_stream_width()
        nelems = np.prod(oshape)
        nbits = nelems * itype_bits
        assert (
            nbits % stream_width == 0
        ), "ACCL: total transfer size must be word multiple"
        ovalues = nbits // stream_width
        return ovalues

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape."
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("dataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = []

    def defines(self, mode):
        # Do the includes here as well as they have dependencies on the defines
        self.code_gen_dict["$DEFINES$"] = []
        if mode == 'cppsim':
            self.code_gen_dict["$DEFINES$"] += [
                "#define CPPSIM",
                '#include "cclo_bfm.h"',
                '#include "accl/sim.hpp"',
            ]
        elif mode == 'ipgen':
            self.code_gen_dict["$DEFINES$"] += [
                '#define ACCL_SYNTHESIS',
            ]

        self.code_gen_dict["$DEFINES$"] += [
            '#include <accl_hls.h>',
            '#include "accl/funcs.hpp"',
        ]

    def get_stream_width(self):
        tbits = self.get_input_datatype().bitwidth()
        return tbits * self.get_nodeattr("NumChannels")

    def verify_node(self):
        ...

class ACCLOut(ACCLOp):
    def get_instream_width(self, ind=0):
        return self.get_stream_width()

    def get_outstream_width(self, ind=0):
        return accl_word_size

    def get_folded_input_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))

        ich_bits = ich * self.get_input_datatype().bitwidth()
        fold = int(math.ceil(ich_bits / accl_word_size))

        return (*vecs, fold, ich)

    def get_folded_output_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))

        num_bits = np.prod(vecs) * ich * self.get_input_datatype().bitwidth()
        fold = int(math.ceil(num_bits / accl_word_size))

        return (fold, 1)

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            '#pragma HLS INTERFACE axis port=cmd_to_cclo',
            '#pragma HLS INTERFACE axis port=sts_from_cclo',
            '#pragma HLS INTERFACE axis port=data_to_cclo',
            '#pragma HLS INTERFACE axis port=in0_{}'.format(self.hls_sname()),
            "#pragma HLS INTERFACE s_axilite port=dpcfg_adr bundle=control",
            "#pragma HLS INTERFACE s_axilite port=comm_adr bundle=control",
        ]

        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

    def strm_decl(self):
        start_port = self.get_nodeattr("startPort")
        rank = self.get_nodeattr("device_id")
        world_size = self.get_nodeattr("worldSize")

        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            'hlslib::Stream<command_word> cmd_to_cclo("cmd_to_cclo"), sts_from_cclo("sts_from_cclo");',
            'hlslib::Stream<stream_word, 512> data_from_cclo("data_from_cclo"), data_to_cclo("data_to_cclo");',
            'hls::stream<ap_uint<{}>> in0_{};'.format(self.get_stream_width(), self.hls_sname()),
            'std::unique_ptr<ACCL::ACCL> accl = init_accl({}, {}, {});'.format(world_size, rank, start_port),
            'std::unique_ptr<CCLO_BFM> cclo = init_cclo_and_wait_for_input({}, {}, {}, cmd_to_cclo, sts_from_cclo, data_from_cclo, data_to_cclo);'.format(start_port, rank, world_size),
            'ap_uint<32> comm_adr = accl->get_communicator_addr();',
            'ap_uint<32> dpcfg_adr = accl->get_arithmetic_config_addr({ACCL::dataType::int32, ACCL::dataType::int32});',
            'bool wait_for_ack = true;',
        ]

    def docompute(self):
        stream_width = self.get_instream_width()

        itype_bits = self.get_input_datatype().bitwidth()
        shape = self.get_folded_input_shape()
        num_bits = np.prod(shape) * itype_bits

        step = math.gcd(stream_width, accl_word_size)

        dest = self.get_nodeattr("otherRank")

        self.code_gen_dict["$DOCOMPUTE$"] = [
            '''accl_out<{}, {}, {}>(
                {},
                comm_adr,
                dpcfg_adr,
                cmd_to_cclo,
                sts_from_cclo,
                data_to_cclo,
                in0_{},
                wait_for_ack
            );'''.format(stream_width, num_bits, step, dest, self.hls_sname()),
            '''
            #ifdef CPPSIM
            cclo->stop();
            #endif
            ''',
        ]

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        if mode != "cppsim":
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim")""".format(
                    mode
                )
            )

        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")

        assert (
            str(context[node.input[0]].dtype) == "float32"
        ), """Input datatype is
        not float32 as expected."""
        expected_inp_shape = self.get_folded_input_shape()

        reshaped_input = context[node.input[0]].reshape(expected_inp_shape)
        if self.get_input_datatype() == DataType["BIPOLAR"]:
            # store bipolar activations as binary
            reshaped_input = (reshaped_input + 1) / 2
            export_idt = DataType["BINARY"]
        else:
            export_idt = self.get_input_datatype()
        # make copy before saving the array
        reshaped_input = reshaped_input.copy()
        np.save(
            os.path.join(code_gen_dir, "input.npy"),
            reshaped_input,
        )

        # Execute node in a new thread so execution can continue to the receiving ACCLIn
        # node.
        self.thread = threading.Thread(
            target=self.execute_op,
            args=(self.onnx_node.output[0],)
        )
        self.thread.start()

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = "%s/input.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []

        # note: the innermost dim is reversed for the input
        self.code_gen_dict["$READNPYDATA$"] += [
            'npy2apintstream<%s, %s, %d, %s>("%s", in0_%s, false);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in, self.hls_sname()),
        ]

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def dataoutstrm(self):
        self.code_gen_dict["$DATAOUTSTREAM$"] = ['']

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            '''void {}(
                STREAM<command_word> &cmd_to_cclo,
                STREAM<command_word> &sts_from_cclo,
                STREAM<stream_word> &data_to_cclo,
                ap_uint<32> comm_adr,
                ap_uint<32> dpcfg_adr,
                hls::stream<ap_uint<{}>> &in0_{},
                bool wait_for_ack
            )'''
            .format(
                self.onnx_node.name,
                self.get_instream_width(),
                self.hls_sname()
            )
        ]

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()

        intf_names["m_axis"] = [("data_to_cclo", accl_word_size), ("cmd_to_cclo", 32)]
        intf_names["s_axis"].append(("sts_from_cclo", 32))
        intf_names["axilite"] = ["s_axi_control"]

        return intf_names

class ACCLIn(ACCLOp):
    def get_instream_width(self, ind=0):
        return accl_word_size

    def get_outstream_width(self, ind=0):
        return self.get_stream_width()

    def get_folded_input_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))

        num_bits = np.prod(vecs) * ich * self.get_input_datatype().bitwidth()
        fold = int(math.ceil(num_bits / accl_word_size))

        return (fold, 1)

    def get_folded_output_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))

        ich_bits = ich * self.get_input_datatype().bitwidth()
        fold = int(math.ceil(ich_bits / accl_word_size))

        return (*vecs, fold, ich)

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            '#pragma HLS INTERFACE axis port=data_from_cclo',
            '#pragma HLS INTERFACE axis port=out_{}'.format(self.hls_sname()),
        ]

        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

    def strm_decl(self):
        start_port = self.get_nodeattr("startPort")
        rank = self.get_nodeattr("device_id")
        world_size = self.get_nodeattr("worldSize")

        assert world_size != 0

        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            'hlslib::Stream<command_word> cmd_to_cclo("cmd_to_cclo"), sts_from_cclo("sts_from_cclo");',
            'hlslib::Stream<stream_word, 512> data_from_cclo("data_from_cclo"), data_to_cclo("data_to_cclo");',
            'hls::stream<ap_uint<{}>> out_{};'.format(self.get_stream_width(), self.hls_sname()),
            'std::unique_ptr<CCLO_BFM> cclo = init_cclo_and_wait_for_input({}, {}, {}, cmd_to_cclo, sts_from_cclo, data_from_cclo, data_to_cclo);'.format(start_port, rank, world_size),
        ]

    def docompute(self):
        stream_width = self.get_stream_width()

        itype_bits = self.get_input_datatype().bitwidth()
        shape = self.get_folded_output_shape()
        num_bits = np.prod(shape) * itype_bits

        step = math.gcd(stream_width, accl_word_size)

        source = self.get_nodeattr("otherRank")

        self.code_gen_dict["$DOCOMPUTE$"] = [
            'accl_in<{}, {}, {}>({}, data_from_cclo, out_{});'.format(
                stream_width,
                num_bits,
                step,
                source,
                self.hls_sname()
            ),
            '''
            #ifdef CPPSIM
            cclo->stop();
            #endif
            ''',
        ]

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        if mode != "cppsim":
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim")""".format(
                    mode
                )
            )

        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")

        self.execute_op(self.onnx_node.input[0])

        super().npy_to_dynamic_output(context)

        if self.get_output_datatype() == DataType["BIPOLAR"]:
            out = context[node.output[0]]
            out = 2 * out - 1
            context[node.output[0]] = out
        oshape = self.get_normal_output_shape()

        assert (
            context[node.output[0]].shape == oshape
        ), """Output shape is not as expected"""

    def read_npy_data(self):
        self.code_gen_dict["$READNPYDATA$"] = ['']

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = ['']

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output.npy" % code_gen_dir
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out_%s, %s, "%s", false);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                self.hls_sname(),
                shape_cpp_str,
                npy_out,
            ),
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            'void {}(STREAM<stream_word> &data_from_cclo, hls::stream<ap_uint<{}>> &out_{})'
            .format(
                self.onnx_node.name,
                self.get_outstream_width(),
                self.hls_sname()
            )
        ]

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()

        intf_names["s_axis"] = [("data_from_cclo", accl_word_size)]

        return intf_names

