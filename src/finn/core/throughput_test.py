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
import subprocess
import numpy as np
import warnings
from finn.util.basic import gen_finn_dt_tensor
from finn.core.rtlsim_exec import rtlsim_exec


def throughput_test_remote(model, batchsize=1000):
    """Runs the throughput test for the given model remotely on the pynq board.
    The metadata properties related to the pynq board have to be set.
    Returns a dictionary with results of the throughput test. Returns None
    if the test fails."""

    pynq_ip = model.get_metadata_prop("pynq_ip")
    pynq_port = int(model.get_metadata_prop("pynq_port"))
    pynq_username = model.get_metadata_prop("pynq_username")
    pynq_password = model.get_metadata_prop("pynq_password")
    pynq_target_dir = model.get_metadata_prop("pynq_target_dir")
    deployment_dir = model.get_metadata_prop("pynq_deploy_dir")
    # extracting last folder of absolute path (deployment_dir)
    deployment_folder = os.path.basename(os.path.normpath(deployment_dir))
    platform = model.get_metadata_prop("platform")
    assert platform in ["alveo", "zynq", "zynq-iodma"]
    bitfile = model.get_metadata_prop("bitfile")
    bitfile = os.path.basename(bitfile)
    if pynq_password == "":
        if "zynq" in platform:
            raise Exception("PYNQ board remote exec needs password for sudo")
        else:
            local_prefix = ""  # assume we are using an ssh key
            warnings.warn("Empty password, make sure you've set up an ssh key")
    else:
        local_prefix = "sshpass -p %s " % pynq_password

    if platform == "alveo":
        # Alveo can run without sudo but needs correct environment
        remote_prefix = "conda activate finn-pynq-alveo; "
    elif "zynq" in platform:
        # PYNQ Zynq boards need to execute with sudo
        remote_prefix = "echo %s | sudo -S " % pynq_password

    # use platform attribute for correct remote execution
    if platform == "alveo":
        remote_cmd = "bash -ic 'bash alveo_run.sh throughput_test %d'" % batchsize
    else:
        remote_cmd = (
            "python3.6 driver.py --exec_mode=throughput_test --batchsize={} "
            "--bitfile={} --inputfile=input.npy --outputfile=output.npy "
            '--platform={} "'
        ).format(batchsize, bitfile, platform)
    cmd = (
        local_prefix + "ssh {}@{} -p {} " '"cd {}/{}; ' + remote_prefix + remote_cmd
    ).format(pynq_username, pynq_ip, pynq_port, pynq_target_dir, deployment_folder)
    bash_command = ["/bin/bash", "-c", cmd]
    process_throughput_test = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    process_throughput_test.communicate()

    # remove any pre-existing metrics file
    try:
        os.remove("{}/nw_metrics.txt".format(deployment_dir))
    except FileNotFoundError:
        pass

    cmd = local_prefix + "scp -P{} {}@{}:{}/{}/nw_metrics.txt {}".format(
        pynq_port,
        pynq_username,
        pynq_ip,
        pynq_target_dir,
        deployment_folder,
        deployment_dir,
    )
    bash_command = ["/bin/bash", "-c", cmd]
    process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    process_compile.communicate()

    try:
        with open("{}/nw_metrics.txt".format(deployment_dir), "r") as file:
            res = eval(file.read())
        return res
    except FileNotFoundError:
        return None


def throughput_test_rtlsim(model, batchsize=100):
    """Runs a throughput test for the given IP-stitched model. When combined
    with tracing, useful to determine bottlenecks and required FIFO sizes."""

    assert (
        model.get_metadata_prop("exec_mode") == "rtlsim"
    ), """Top-level exec_mode
    metadata_prop must be set to rtlsim"""

    # create random input
    iname = model.graph.input[0].name
    ishape = model.get_tensor_shape(iname)
    ishape_batch = ishape
    ishape_batch[0] = batchsize
    idt = model.get_tensor_datatype(iname)
    dummy_input = gen_finn_dt_tensor(idt, ishape_batch)
    # compute input/output sizes
    oname = model.graph.output[0].name
    oshape = model.get_tensor_shape(oname)
    oshape_batch = oshape
    oshape_batch[0] = batchsize
    odt = model.get_tensor_datatype(oname)
    i_bytes = (np.prod(ishape_batch) * idt.bitwidth()) / 8
    o_bytes = (np.prod(oshape_batch) * odt.bitwidth()) / 8
    # make empty exec context and insert input
    ctx = model.make_empty_exec_context()
    ctx[iname] = dummy_input
    # remove liveness threshold, launch rtlsim
    os.environ["LIVENESS_THRESHOLD"] = "-1"
    rtlsim_exec(model, ctx)
    # extract metrics
    cycles = int(model.get_metadata_prop("cycles_rtlsim"))
    clk_ns = float(model.get_metadata_prop("clk_ns"))
    fclk_mhz = 1 / (clk_ns * 0.001)
    runtime_s = (cycles * clk_ns) * (10 ** -9)
    res = dict()
    res["cycles"] = cycles
    res["runtime[ms]"] = runtime_s * 1000
    res["throughput[images/s]"] = batchsize / runtime_s
    res["DRAM_in_bandwidth[Mb/s]"] = i_bytes * 0.000001 / runtime_s
    res["DRAM_out_bandwidth[Mb/s]"] = o_bytes * 0.000001 / runtime_s
    res["fclk[mhz]"] = fclk_mhz
    res["N"] = batchsize

    return res
