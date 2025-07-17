from dataclasses import dataclass, fields
from pathlib import Path
from typing import Tuple, FrozenSet, Callable, Dict, List, Optional
from qonnx.util.basic import roundup_to_integer_multiple

from finn.util.context import Context

import numpy as np
import os
import warnings
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.basic import get_liveness_threshold_cycles
import finn_xsi.adapter as finnxsi

@dataclass
class KernelProjection:
    """ resource and performance projections of a given kernel instance """
    cycles : int
    LUTs   : int
    DSPs   : int
    BRAMs  : int

class KernelInvalidParameter(Exception):
    def __init__(self, message: str):  
        super().__init__(message)  
        self.message = message  

    def __str__(self):  
        return f"Kernel Badly Configured: {self.message}"  

@dataclass(frozen=True, init=False)
class Kernel:
    """ Abstract class for a Kernel """

    def __init__(self, **kwargs):
        self._attribute_init(**kwargs)
        self.__post_init__()
        object.__setattr__(self, "subkernels", self._init_subkernels(**kwargs))

    def __post_init__(self) -> None:  

        # Ensure impl_style is specified correctly.
        if not hasattr(self, "impl_style"):
            raise RuntimeError(f"The implementation style (impl_style) for the kernel was not specified, this needs to be either 'hls' or 'rtl'")
        if hasattr(self,"impl_style"):
            if self.impl_style not in ["rtl", "hls", "sip"]:
                raise RuntimeError(f"Error impl_style must be 'hls', 'rtl' or 'sip' NOT {self.impl_style}")

        # Check constraints if optional constraints field is specified.
        if hasattr(self, "_constraints"):
            # Loose "Type check" for _constraints list
            if not isinstance(self._constraints, Tuple) or not all(callable(c) for c in self._constraints):
                raise TypeError("The '_constraints' field must be a list[Callable[['Kernel'], bool]]")

            # Check the constraints of the subclass to see if it is viable.
            for index, expr in enumerate(self._constraints):  
                result = expr(self)  
                if not result:
                    raise KernelInvalidParameter(f"Kernel Instantiation failed assertion check {index=}")

    ######################### Common kernel attributes #########################
    name: str
    len_node_input: int
    len_node_output: int

    # Initialise attributes from kwargs, subkernels can override this for other behaviour.
    def _attribute_init(self, **kwargs) -> None:
        """ Discard extra keyword arguments. """
        ignore_fields = {'_constraints', 'impl_style', 'sharedFiles', 'kernelFiles', 'subkernels'}
        valid_fields = {f.name for f in fields(self.__class__)} - ignore_fields
        for field_name in valid_fields:
            if field_name in kwargs:
                object.__setattr__(self, field_name, kwargs.pop(field_name))

    ######################### Constraints #########################
    _constraints: Optional[Tuple[Callable[['Kernel'],bool]]]

    ######################### Implementation style, rtl/hls/sip #########################
    impl_style: str

    ######################### Code Generation #########################
    # Set of library names to resolve and include if HLS
    # Set of file paths to copy over as kernel files if RTL
    sharedFiles: FrozenSet[Tuple[str,Path]] = frozenset()

    # Paths of directories to be included for HLS synth if HLS, relative to finn
    # Paths to shared RTL files if RTL, relative to finn
    kernelFiles: FrozenSet[Path] = frozenset()

    # Tuples of [instance file generators, path to generated file relative to instance dir]
    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:
        return frozenset()

    # This method is called by code builders.
    def generate_instance_files(self, ctx: Context) -> None:
        for generator, _ in self.instanceFiles:
            generator(ctx)

    def get_abs_verilog_files(self, node_ctx: Context) -> set[Path]:
        """Return list of all Verilog files used for this node, abs paths."""

        if self.impl_style == "hls":
            verilog_paths: set[Path] = set()
            verilog_path = Path("{}/project_{}/sol1/impl/verilog/".format(node_ctx.directory, self.name))
            # default impl only returns the HLS verilog codegen dir and subcore (impl/ip/hdl/ip) dir if it exists
            # TODO: Might be able to remove subcore_verilog_path after splitting MVAU-like into SIP kernels?
            verilog_paths.add(verilog_path)

            verilog_files = set()
            for verilog_path in verilog_paths:
                for f in verilog_path.iterdir():
                    if f.suffix == ".v":
                        verilog_files.add(f)
            return verilog_files

        elif self.impl_style == "rtl":
            rtl_suffixes = [".v", ".sv", ".vh"]
            code_gen_dir = node_ctx.directory
            kernel_dir = node_ctx.top_ctx.get_kernel_dir(type(self).__name__)
            shared_dir = node_ctx.top_ctx.shared_dir
            # Get paths to instance files.
            instance_verilog_files = {file for file in code_gen_dir.rglob('*') if (file.is_file() and file.suffix in rtl_suffixes)}
            # Get paths to kernel files.
            kernel_verilog_files = {file for file in kernel_dir.rglob('*') if (file.is_file() and file.suffix in rtl_suffixes)}
            # Get paths to shared files.
            shared_verilog_files = set()
            for shared_path in node_ctx.shared:
                if (shared_path.is_file() and shared_path.suffix in rtl_suffixes):
                    shared_verilog_files.add(shared_dir / shared_path.name)
                else:
                    shared_verilog_files |= {shared_dir / file.name for file in shared_path.rglob('*') if (file.is_file() and file.suffix in rtl_suffixes)}
            return instance_verilog_files | kernel_verilog_files | shared_verilog_files

        elif self.impl_style == "sip":
            rtl_suffixes = [".v", ".sv", ".vh"]
            code_gen_dir = node_ctx.directory
            # Get paths to instance files.
            instance_verilog_files = {file for file in code_gen_dir.iterdir() if (file.is_file() and file.suffix in rtl_suffixes)}

            verilog_files = set()
            for subkernel in self.subkernels:
                verilog_files |= subkernel.get_abs_verilog_files(node_ctx.get_subcontext(subkernel.name))
            return instance_verilog_files | verilog_files

        else:
            raise RuntimeError(f"Instance {self.name} of kernel {type(self).__name__} has unknown impl_style {self.impl_style}.")

    ######################### Stitched kernel things #########################
    # Keep a list of kernels that have to be generated as subnodes.
    # The __init__ method calls _init_subkernels and assigns its output to subkernels.
    subkernels: Tuple["Kernel"]

    def _init_subkernels(self, **kwargs) -> Tuple["Kernel"]:
        return tuple()

    ######################### Projections #########################
    def projection(self) -> KernelProjection:
        """ Returns a projection of the configured kernels performance across various metrics """
        return KernelProjection(cycles=None, LUTs=None, DSPs=None, BRAMs=None)

    ######################### RTL Simulation #########################
    def get_rtlsim(self, code_gen_dir, rtlsim_trace):
        """Return a xsi wrapper for the emulation library
        for this node."""

        rtlsim_so = str(Path(code_gen_dir) / Path(f"xsim.dir/{self.name}/xsimk.so"))
        assert os.path.isfile(rtlsim_so), "Cannot find rtlsim library."

        sim_base, sim_rel = rtlsim_so.split("xsim.dir")
        sim_rel = "xsim.dir" + sim_rel
        # pass in correct tracefile from attribute
        tracefile = rtlsim_trace
        if tracefile == "default":
            tracefile = self.name + ".wdb"
        sim = finnxsi.load_sim_obj(str(Path.cwd() / Path(sim_base)), sim_rel, tracefile)

        return sim

    def rtlsim_multi_io(self, sim, io_dict, node):
        inst = getCustomOp(node)
        "Run rtlsim for this node, supports multiple i/o streams."
        num_out_values = inst.get_number_output_values()
        total_cycle_count = finnxsi.rtlsim_multi_io(
            sim,
            io_dict,
            num_out_values,
            sname="_V_",
            liveness_threshold=get_liveness_threshold_cycles(),
        )

        inst.set_nodeattr("cycles_rtlsim", total_cycle_count)

    def reset_rtlsim(self, sim):
        """Sets reset input in finnxsi to zero, toggles the clock and set it
        back to one"""
        finnxsi.reset_rtlsim(sim)

    def close_rtlsim(self, sim):
        "Close and free up resources for rtlsim."
        finnxsi.close_rtlsim(sim)

    def build_rtlsim(self, node_ctx: Context, rtlsim_dir: Path, rtlsim_trace: str) -> None:

        verilog_files = self.get_abs_verilog_files(node_ctx)

        verilog_files = [Path("../..") / file.relative_to(node_ctx.top_ctx.directory.parent) for file in verilog_files]

        single_src_dir = rtlsim_dir / Path("rtlsim_" + self.name + "_")
        single_src_dir.mkdir(exist_ok=True)
        trace_file = rtlsim_trace
        debug = not (trace_file is None or trace_file == "")
        ret = finnxsi.compile_sim_obj(
            self.name, [str(file) for file in verilog_files], str(single_src_dir), debug
        )

    def execute_rtlsim(self, context, graph, code_gen_dir, node, rtlsim_trace):
        inst = getCustomOp(node)
        inputs = {}
        for i, inp in enumerate(node.input):
            exp_ishape = tuple(self.get_normal_input_shape(i))
            folded_ishape = inst.get_folded_input_shape(i)
            inp_val = context[inp]

            if self.impl_style == 'rtl':
                assert str(inp_val.dtype) == "float32", "Input datatype is not float32"
            elif self.impl_style == 'hls':
                # Make sure the input has the right container datatype
                if inp_val.dtype is not np.float32:
                    # Issue a warning to make the user aware of this type-cast
                    warnings.warn(
                        f"{self.name}: Changing input container datatype from "
                        f"{inp_val.dtype} to {np.float32}"
                    )
                    # Convert the input to floating point representation as the
                    # container datatype
                    inp_val = inp_val.astype(np.float32)

            assert inp_val.shape == exp_ishape, "Input shape doesn't match expected shape."
            export_idt = inst.get_input_datatype(i)

            if self.impl_style == 'hls':
                if export_idt == DataType["BIPOLAR"]:
                    # store bipolar activations as binary
                    inp_val = (inp_val + 1) / 2
                    export_idt = DataType["BINARY"]

            reshaped_input = inp_val.reshape(folded_ishape)
            np.save(os.path.join(code_gen_dir, "input_%s.npy" % i), reshaped_input)
            nbits = inst.get_instream_width(i)

            if self.impl_style == 'hls':
                # if the stream is not exposed, it has 0 width and no npy file will be created
                if nbits == 0:
                    continue

            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_{}.npy".format(code_gen_dir, i), export_idt, nbits
            )
            inputs["in%s" % i] = rtlsim_inp
        outputs = {}
        for o, outp in enumerate(node.output):
            outputs["out%s" % o] = []
        # assembled execution context
        io_dict = {"inputs": inputs, "outputs": outputs}

        sim = self.get_rtlsim(code_gen_dir, rtlsim_trace)
        self.reset_rtlsim(sim)
        self.rtlsim_multi_io(sim, io_dict, node)
        self.close_rtlsim(sim)
        for o, outp in enumerate(node.output):
            rtlsim_output = io_dict["outputs"]["out%s" % o]
            odt = inst.get_output_datatype(o)
            target_bits = odt.bitwidth()
            packed_bits = inst.get_outstream_width(o)

            if self.impl_style == 'rtl':
                out_npy_path = "{}/output.npy".format(code_gen_dir)
            elif self.impl_style == 'hls':
                out_npy_path = "{}/output_{}.npy".format(code_gen_dir, o)

            out_shape = inst.get_folded_output_shape(o)
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )
            # load and reshape output
            exp_oshape = tuple(self.get_normal_output_shape(o))
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[outp] = output

            assert (
                context[outp].shape == exp_oshape
            ), "Output shape doesn't match expected shape."

    ######################### Other common methods #########################
    def get_verilog_top_module_intf_names(self) -> Dict[str,List]:
        """Return a dict of names of input and output interfaces.
        The keys reflect the protocols each interface implements:
        'clk', 'rst', 'm_axis', 's_axis', 'aximm', 'axilite'.
        Values are lists of tuples (axis, aximm) or names (axilite):
        'axis' tuples correspond to the list of node inputs in order,
        each tuple is (interface_name, interface_width_bits).
        axilite always assumed to be 32 bits and is not tuple (name only).
        Each block must have at most one aximm and one axilite."""
        intf_names = {}
        intf_names["clk"] = ["ap_clk"]
        intf_names["rst"] = ["ap_rst_n"]
        intf_names["s_axis"] = []
        for i in range(self.len_node_input):
            # not every node input will result in an interface of the produced HW
            # filter out inputs that have no stream width associated with them
            width = self.get_instream_width_padded(i)
            if width != 0:
                intf_names["s_axis"].append(("in%d_V" % (i), self.get_instream_width_padded(i)))
        intf_names["m_axis"] = []
        for i in range(self.len_node_output):
            intf_names["m_axis"].append(("out%d_V" % (i), self.get_outstream_width_padded(i)))
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        return intf_names

    def get_instream_width_padded(self, ind=0) -> int:
        """Returns input stream width padded to a multiple of 8. This is required
        by the AXI Stream spec."""
        in_width = self.get_instream_width(ind=ind)
        if in_width != 0:
            return roundup_to_integer_multiple(in_width, 8)
        else:
            return 0

    def get_outstream_width_padded(self, ind=0) -> int:
        """Returns output stream width padded to a multiple of 8. This is required
        by the AXI Stream spec."""
        out_width = self.get_outstream_width(ind=ind)
        return roundup_to_integer_multiple(out_width, 8)

    def get_instream_width(self, ind=0) -> int:
        pass

    def get_outstream_width(self, ind=0) -> int:
        pass

    @property
    def hls_sname(self) -> str:
        """Get the naming convention used by Vitis HLS for stream signals
        Example: the TDATA for a stream called "out" would be out_V_TDATA.
        """
        return "V"
