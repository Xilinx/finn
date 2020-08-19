# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import random
import string
import subprocess
import tempfile
import warnings

import numpy as np

from finn.core.datatype import DataType

# mapping from PYNQ board names to FPGA part names
pynq_part_map = dict()
pynq_part_map["Ultra96"] = "xczu3eg-sbva484-1-e"
pynq_part_map["Pynq-Z1"] = "xc7z020clg400-1"
pynq_part_map["Pynq-Z2"] = "xc7z020clg400-1"
pynq_part_map["ZCU104"] = "xczu7ev-ffvc1156-2-e"

# native AXI HP port width (in bits) for PYNQ boards
pynq_native_port_width = dict()
pynq_native_port_width["Pynq-Z1"] = 64
pynq_native_port_width["Pynq-Z2"] = 64
pynq_native_port_width["Ultra96"] = 128
pynq_native_port_width["ZCU104"] = 128

# Alveo device and platform mappings
alveo_part_map = dict()
alveo_part_map["U50"] = "xcu50-fsvh2104-2L-e"
alveo_part_map["U200"] = "xcu200-fsgd2104-2-e"
alveo_part_map["U250"] = "xcu250-figd2104-2L-e"
alveo_part_map["U280"] = "xcu280-fsvh2892-2L-e"

alveo_default_platform = dict()
alveo_default_platform["U50"] = "xilinx_u50_gen3x16_xdma_201920_3"
alveo_default_platform["U200"] = "xilinx_u200_xdma_201830_2"
alveo_default_platform["U250"] = "xilinx_u250_xdma_201830_2"
alveo_default_platform["U280"] = "xilinx_u280_xdma_201920_3"

# resource availability for PYNQ and Alveo boards
platform_resource_counts = dict()
platform_resource_counts["Pynq-Z1"] = {
    "slr0": {"LUT": 53200, "BRAM_18K": 280, "DSP": 220, "URAM": 0}
}
platform_resource_counts["Pynq-Z2"] = {
    "slr0": {"LUT": 53200, "BRAM_18K": 280, "DSP": 220, "URAM": 0}
}
platform_resource_counts["Ultra96"] = {
    "slr0": {"LUT": 71000, "BRAM_18K": 412, "DSP": 360, "URAM": 0}
}
platform_resource_counts["ZCU104"] = {
    "slr0": {"LUT": 230000, "BRAM_18K": 610, "DSP": 1728, "URAM": 92}
}
platform_resource_counts["U50"] = {
    "slr0": {"LUT": 369000, "BRAM_18K": 1128, "DSP": 2580, "URAM": 304},
    "slr1": {"LUT": 362000, "BRAM_18K": 1128, "DSP": 2760, "URAM": 304},
}
platform_resource_counts["U200"] = {
    "slr0": {"LUT": 355000, "BRAM_18K": 1276, "DSP": 2265, "URAM": 320},
    "slr1": {"LUT": 160000, "BRAM_18K": 652, "DSP": 1317, "URAM": 160},
    "slr2": {"LUT": 355000, "BRAM_18K": 1276, "DSP": 2265, "URAM": 320},
}
platform_resource_counts["U250"] = {
    "slr0": {"LUT": 345000, "BRAM_18K": 1000, "DSP": 2877, "URAM": 320},
    "slr1": {"LUT": 345000, "BRAM_18K": 1000, "DSP": 2877, "URAM": 320},
    "slr2": {"LUT": 345000, "BRAM_18K": 1000, "DSP": 2877, "URAM": 320},
    "slr3": {"LUT": 345000, "BRAM_18K": 1000, "DSP": 2877, "URAM": 320},
}
platform_resource_counts["U280"] = {
    "slr0": {"LUT": 369000, "BRAM_18K": 1014, "DSP": 2733, "URAM": 320},
    "slr1": {"LUT": 333000, "BRAM_18K": 936, "DSP": 2877, "URAM": 320},
    "slr2": {"LUT": 367000, "BRAM_18K": 1024, "DSP": 2880, "URAM": 320},
}


def get_rtlsim_trace_depth():
    """Return the trace depth for rtlsim via PyVerilator. Controllable
    via the RTLSIM_TRACE_DEPTH environment variable. If the env.var. is
    undefined, the default value of 1 is returned. A trace depth of 1
    will only show top-level signals and yield smaller .vcd files.

    The following depth values are of interest for whole-network stitched IP
    rtlsim:
    - level 1 shows top-level input/output streams
    - level 2 shows per-layer input/output streams
    - level 3 shows per full-layer I/O including FIFO count signals
    """

    try:
        return int(os.environ["RTLSIM_TRACE_DEPTH"])
    except KeyError:
        return 1


def get_remote_vivado():
    """Return the address of the remote Vivado synthesis server as set by the,
    REMOTE_VIVADO environment variable, otherwise return None"""

    try:
        return os.environ["REMOTE_VIVADO"]
    except KeyError:
        return None


def get_num_default_workers():
    """Return the number of workers for parallel transformations. Controllable
    via the NUM_DEFAULT_WORKERS environment variable. If the env.var. is
    undefined, the default value of 1 is returned.
    """

    try:
        return int(os.environ["NUM_DEFAULT_WORKERS"])
    except KeyError:
        return 1


def get_finn_root():
    "Return the root directory that FINN is cloned into."

    try:
        return os.environ["FINN_ROOT"]
    except KeyError:
        raise Exception(
            """Environment variable FINN_ROOT must be set
        correctly. Please ensure you have launched the Docker contaier correctly.
        """
        )


def get_execution_error_thresh():
    "Return the max error that is allowed for rounding in FINN execution."
    try:
        return float(os.environ["ERROR_THRESH"])
    except KeyError:
        return 1e-2


def get_sanitize_quant_tensors():
    """Return whether tensors with quantization annotations should be sanitized.
    Enabled by default, disabling will yield faster ONNX execution but may give
    incorrect results. Use with caution."""
    try:
        return int(os.environ["SANITIZE_QUANT_TENSORS"])
    except KeyError:
        # enabled by default
        return 1


def make_build_dir(prefix=""):
    """Creates a temporary folder with given prefix to be used as a build dir.
    Use this function instead of tempfile.mkdtemp to ensure any generated files
    will survive on the host after the FINN Docker container exits."""
    try:
        inst_prefix = os.environ["FINN_INST_NAME"] + "/"
        return tempfile.mkdtemp(prefix=inst_prefix + prefix)
    except KeyError:
        raise Exception(
            """Environment variable FINN_INST_NAME must be set
        correctly. Please ensure you have launched the Docker contaier correctly.
        """
        )


def get_by_name(container, name, name_field="name"):
    """Return item from container by .name field if it exists, None otherwise"""
    names = [getattr(x, name_field) for x in container]
    try:
        ind = names.index(name)
        return container[ind]
    except ValueError:
        return None


def remove_by_name(container, name, name_field="name"):
    """Remove item from container by .name field if it exists."""
    item = get_by_name(container, name, name_field)
    if item is not None:
        container.remove(item)


def random_string(stringLength=6):
    """Randomly generate a string of letters and digits."""
    lettersAndDigits = string.ascii_letters + string.digits
    return "".join(random.choice(lettersAndDigits) for i in range(stringLength))


def interleave_matrix_outer_dim_from_partitions(matrix, n_partitions):
    """Interleave the outermost dimension of a matrix from given
    partitions (n_partitions)."""
    if type(matrix) != np.ndarray or matrix.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        matrix = np.asarray(matrix, dtype=np.float32)
    shp = matrix.shape
    ndim = matrix.ndim
    # ensure # partitions evenly divide the outermost dimension
    assert (
        shp[0] % n_partitions == 0
    ), """The outermost dimension is not divisable
    by the number of partitions."""
    # only tested for matrices
    assert (
        ndim == 2
    ), """The dimension of the matrix is not 2. Currently this function
    only works for matrices."""
    # interleave rows between PEs using reshape + transpose
    matrix_r = matrix.reshape(-1, n_partitions, shp[1]).transpose((1, 0, 2))
    matrix_r = matrix_r.reshape(n_partitions, -1, shp[1])
    return matrix_r


def roundup_to_integer_multiple(x, factor):
    """Round up integer x to the nearest integer multiple of integer factor.
    Returns x if factor is set to -1. Both x and factor must otherwise be
    positive."""
    # ensure integers
    assert int(x) == x, "The input x is not an integer."
    assert int(factor) == factor, "The input factor is not an integer."
    # use -1 to indicate no padding needed
    if factor == -1:
        return x
    # ensure positive values
    assert factor > 0 and x > 0, "Factor and x are <= 0."
    if x < factor:
        return factor
    else:
        if x % factor == 0:
            return x
        else:
            return x + (factor - (x % factor))


def pad_tensor_to_multiple_of(ndarray, pad_to_dims, val=0, distr_pad=False):
    """Pad each dimension of given NumPy ndarray using val, so that each
    dimension is a multiple of the respective value in pad_to_dims. -1 means
    do not pad that particular dimension. If distr_pad is False, all padding
    will be inserted after the existing values; otherwise it will be split
    evenly between before and after the existing values, with one extra value
    inserted after if the padding amount is not divisible by two."""
    if type(ndarray) != np.ndarray or ndarray.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        ndarray = np.asarray(ndarray, dtype=np.float32)
    assert ndarray.ndim == len(
        pad_to_dims
    ), """The dimensions of the input
    array don't match the length of the pad_to_dims value."""
    # compute the desired shape
    desired = zip(list(ndarray.shape), list(pad_to_dims))
    desired = map(lambda x: roundup_to_integer_multiple(x[0], x[1]), desired)
    desired = np.asarray(list(desired), dtype=np.int32)
    current = np.asarray(ndarray.shape, dtype=np.int32)
    pad_amt = desired - current
    # add padding to get to the desired shape
    if distr_pad:
        pad_before = (pad_amt // 2).astype(np.int32)
        pad_after = pad_amt - pad_before
        pad_amt = list(zip(pad_before, pad_after))
    else:
        # all padding is added after the existing values
        pad_amt = list(map(lambda x: (0, x), pad_amt))
    ret = np.pad(ndarray, pad_amt, mode="constant", constant_values=val)
    assert (
        np.asarray(ret.shape, dtype=np.int32) == desired
    ).all(), """The
    calculated output array doesn't match the desired/expected one."""
    return ret


def gen_finn_dt_tensor(finn_dt, tensor_shape):
    """Generates random tensor in given shape and with given FINN DataType."""
    if type(tensor_shape) == list:
        tensor_shape = tuple(tensor_shape)
    if finn_dt == DataType.BIPOLAR:
        tensor_values = np.random.randint(2, size=tensor_shape)
        tensor_values = 2 * tensor_values - 1
    elif finn_dt == DataType.BINARY:
        tensor_values = np.random.randint(2, size=tensor_shape)
    elif "INT" in finn_dt.name or finn_dt == DataType.TERNARY:
        tensor_values = np.random.randint(
            finn_dt.min(), high=finn_dt.max() + 1, size=tensor_shape
        )
    else:
        raise ValueError(
            "Datatype {} is not supported, no tensor could be generated".format(finn_dt)
        )
    # always use float type as container
    return tensor_values.astype(np.float32)


def calculate_signed_dot_prod_range(dt_a, dt_b, len):
    """Returns the (min,max) values a dot product between two signed vectors of
    types dt_a and dt_b of len elements can take."""
    assert (
        dt_a.signed() and dt_b.signed()
    ), """The input values are not both
    signed vectors."""
    min_prod = 2 ** 30
    max_prod = -(2 ** 30)
    for a_val in [dt_a.min(), dt_a.max()]:
        for b_val in [dt_b.min(), dt_b.max()]:
            prod = a_val * b_val * len
            if prod < min_prod:
                min_prod = prod
            if prod > max_prod:
                max_prod = prod
    return (min_prod, max_prod)


def sanitize_quant_values(model, node_tensors, execution_context, check_values=False):
    """ Sanitize given list of tensors in execution_context by rounding values
    that are supposed to be integers (as indicated by their quantization
    annotation). Will raise an assertion if the amount of rounding is too large.
    Returns the sanitized execution context.

    If check_values is specified, an extra DataType.allowed() check will be
    performed on any rounded tensors.

    Background:
    FINN uses floating point tensors as a carrier data type to represent
    integers. Floating point arithmetic can introduce rounding errors, e.g.
    (int_num * float_scale) / float_scale is not always equal to int_num.
    We use this function to ensure that the values that are supposed to be
    integers are indeed integers.
    """

    for tensor in node_tensors:
        dtype = model.get_tensor_datatype(tensor)
        # floats don't need sanitization, skip to next
        # introduces less quicker runtime
        if dtype == DataType.FLOAT32:
            continue
        current_values = execution_context[tensor]
        updated_values = current_values
        has_to_be_rounded = False
        # TODO: vectorize with numpy
        for value in np.nditer(current_values):
            if not dtype.allowed(value):
                has_to_be_rounded = True
                break
        if has_to_be_rounded:
            updated_values = np.round(current_values)
            warnings.warn(
                "The values of tensor {} can't be represented "
                "with the set FINN datatype ({}), they will be rounded to match the "
                "FINN datatype.".format(tensor, dtype)
            )
        # check if rounded values are not too far from original values
        max_error = max(np.abs(current_values - updated_values).flatten())
        if max_error <= get_execution_error_thresh():
            if check_values is True:
                # check again if values can now be represented with set finn datatype
                # TODO: vectorize with numpy
                for value in np.nditer(updated_values):
                    if not dtype.allowed(value):
                        raise Exception(
                            """Values can't be represented with set
                                finn datatype ({}) for input {}""".format(
                                dtype, tensor
                            )
                        )
            execution_context[tensor] = updated_values
        else:
            raise Exception(
                """Rounding error is too high to match set FINN
            datatype ({}) for input {}""".format(
                    dtype, tensor
                )
            )
    return execution_context


class CppBuilder:
    """Builds the g++ compiler command to produces the executable of the c++ code
    in code_gen_dir which is passed to the function build() of this class."""

    def __init__(self):
        self.include_paths = []
        self.cpp_files = []
        self.executable_path = ""
        self.code_gen_dir = ""
        self.compile_components = []
        self.compile_script = ""

    def append_includes(self, library_path):
        """Adds given library path to include_paths list."""
        self.include_paths.append(library_path)

    def append_sources(self, cpp_file):
        """Adds given c++ file to cpp_files list."""
        self.cpp_files.append(cpp_file)

    def set_executable_path(self, path):
        """Sets member variable "executable_path" to given path."""
        self.executable_path = path

    def build(self, code_gen_dir):
        """Builds the g++ compiler command according to entries in include_paths
        and cpp_files lists. Saves it in bash script in given folder and
        executes it."""
        # raise error if includes are empty
        self.code_gen_dir = code_gen_dir
        self.compile_components.append("g++ -o " + str(self.executable_path))
        for cpp_file in self.cpp_files:
            self.compile_components.append(cpp_file)
        for lib in self.include_paths:
            self.compile_components.append(lib)
        bash_compile = ""
        for component in self.compile_components:
            bash_compile += str(component) + " "
        self.compile_script = str(self.code_gen_dir) + "/compile.sh"
        with open(self.compile_script, "w") as f:
            f.write("#!/bin/bash \n")
            f.write(bash_compile + "\n")
        bash_command = ["bash", self.compile_script]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
