import os
import random
import string
import subprocess
import tempfile

import numpy as np

from finn.core.datatype import DataType

# mapping from PYNQ board names to FPGA part names
pynq_part_map = dict()
pynq_part_map["Ultra96"] = "xczu3eg-sbva484-1-e"
pynq_part_map["Pynq-Z1"] = "xc7z020clg400-1"


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
    """Remove item from container by .name field if it exists"""
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
    assert shp[0] % n_partitions == 0
    # only tested for matrices
    assert ndim == 2
    # interleave rows between PEs using reshape + transpose
    matrix_r = matrix.reshape(-1, n_partitions, shp[1]).transpose((1, 0, 2))
    matrix_r = matrix_r.reshape(n_partitions, -1, shp[1])
    return matrix_r


def roundup_to_integer_multiple(x, factor):
    """Round up integer x to the nearest integer multiple of integer factor.
    Returns x if factor is set to -1. Both x and factor must otherwise be
    positive."""
    # ensure integers
    assert int(x) == x
    assert int(factor) == factor
    # use -1 to indicate no padding needed
    if factor == -1:
        return x
    # ensure positive values
    assert factor > 0 and x > 0
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
    assert ndarray.ndim == len(pad_to_dims)
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
    assert (np.asarray(ret.shape, dtype=np.int32) == desired).all()
    return ret


def gen_finn_dt_tensor(finn_dt, tensor_shape):
    """Generates random tensor in given shape and with given FINN DataType"""
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
    assert dt_a.signed() and dt_b.signed()
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
        and cpp_files lists. Saves it in bash script and executes it."""
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
