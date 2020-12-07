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

# flake8: noqa

# template for the PYNQ shell integration configuration tcl script
ip_config_tcl_template = """
variable config_ip_repo
variable config_ip_vlnv
variable config_ip_bytes_in
variable config_ip_bytes_out
variable config_ip_axis_name_in
variable config_ip_axis_name_out
variable config_ip_use_axilite
variable config_ip_axilite_name
variable config_ip_project_dir
variable config_output_products_dir
variable config_remote_cache
variable config_util_report_filename
variable config_ip_fclk

# for arguments involving paths below: use absolute paths or relative to the
# platform/overlay/bitstream folder
# where to create the project
set config_ip_project_dir %s
# IP repositories that the project depends on
set config_ip_repo %s
# where the produced bitfile and .hwh file will be placed
set config_output_products_dir %s
# where the synth util XML report will be written
set config_util_report_filename %s

# non-path arguments
# VLNV of the IP block
set config_ip_vlnv %s
# width of the AXI stream into the IP, in bytes
set config_ip_bytes_in %d
# width of the AXI stream out of the IP, in bytes
set config_ip_bytes_out %d
# the name of the input AXI stream interface
set config_ip_axis_name_in %s
# the name of the output AXI stream interface
set config_ip_axis_name_out %s
# the name of the clock signal
set config_ip_clk_name %s
# the name of the active-low reset signal
set config_ip_nrst_name %s
# whether the IP needs an AXI Lite interface for control
set config_ip_use_axilite 1
# name of AXI Lite interface
set config_ip_axilite_name %s
# Vivado OOC IP cache
set config_remote_cache "%s"
# clock frequency
set config_ip_fclk %f
"""

call_pynqshell_makefile_template = """
#!/bin/bash
cd %s
export platform=%s
export ip_config=%s
make %s
cd %s
"""

pynq_driver_template = """
import argparse
import os
from pynq import Overlay
import numpy as np
from pynq import allocate
import time
from finn.util.data_packing import (
    finnpy_to_packed_bytearray,
    packed_bytearray_to_finnpy
)
from finn.core.datatype import DataType
from pynq.ps import Clocks

class FINNAccelDriver():
    def __init__(self, N, bitfile, platform="$PLATFORM$"):
        \"\"\"Instantiate the FINN accelerator driver.
        Gets batchsize (N) as integer and path to bitfile as string.\"\"\"
        self.platform = platform
        self.N = N
        # input FINN DataType
        self.idt = $INPUT_FINN_DATATYPE$
        # output FINN DataType
        self.odt = $OUTPUT_FINN_DATATYPE$
        # input and output shapes
        self.ishape_normal = $INPUT_SHAPE_NORMAL$
        self.oshape_normal = $OUTPUT_SHAPE_NORMAL$
        self.ishape_folded = $INPUT_SHAPE_FOLDED$
        self.oshape_folded = $OUTPUT_SHAPE_FOLDED$
        self.ishape_packed = $INPUT_SHAPE_PACKED$   # datatype np.uint8
        self.oshape_packed = $OUTPUT_SHAPE_PACKED$  # datatype np.uint8
        # load bitfile and set up accelerator
        self.ol = Overlay(bitfile)
        # neuron folding factor of output = iterations per sample
        self.itersPerSample = self.oshape_packed[-2]
        # clock frequency as specified by user
        self.fclk_mhz = $CLOCK_FREQ_MHZ$
        if self.platform == "alveo":
            self.idma = self.ol.idma0
            self.odma = self.ol.odma0
        elif self.platform == "zynq-iodma":
            self.idma = self.ol.idma0
            self.odma = self.ol.odma0
            # set the clock frequency as specified by user during transformations
            if self.fclk_mhz > 0:
                Clocks.$CLK_NAME$ = self.fclk_mhz
        else:
            raise ValueError("Supported platforms are zynq-iodma alveo")

        # allocate a PYNQ buffer for the packed input and buffer
        if self.platform == "alveo":
            self.ibuf_packed_device = allocate(shape=self.ishape_packed, dtype=np.uint8)
            self.obuf_packed_device = allocate(shape=self.oshape_packed, dtype=np.uint8)
        else:
            self.ibuf_packed_device = allocate(shape=self.ishape_packed, dtype=np.uint8, cacheable=True)
            self.obuf_packed_device = allocate(shape=self.oshape_packed, dtype=np.uint8, cacheable=True)
        # load any runtime weights
        self.load_runtime_weights()

    def load_runtime_weights(self, flush_accel=True, verify=True):
        w_filenames = []
        runtime_weight_dir = "runtime_weights/"
        for (dirpath, dirnames, filenames) in os.walk(runtime_weight_dir):
            w_filenames.extend(filenames)
        rt_weight_dict = {}
        for w_filename in w_filenames:
            if w_filename.endswith(".dat"):
                with open(runtime_weight_dir + "/" + w_filename, "r") as f:
                    dat = f.read()
            layer_w=np.fromiter([int(x,16) for x in dat.strip().split("\n")], dtype=np.uint32)
            layer_ind=int(w_filename.split("_")[0])
            rt_weight_dict[layer_ind] = layer_w
        for layer_ind in rt_weight_dict.keys():
            cand_if_name = "StreamingDataflowPartition_1/s_axilite_%d" % layer_ind
            if cand_if_name in self.ol.ip_dict.keys():
                layer_mmio = getattr(self.ol.StreamingDataflowPartition_1, "s_axilite_%d" % layer_ind).mmio
                layer_w = rt_weight_dict[layer_ind]
                layer_mmio.write_mm(0, layer_w.tobytes())
                if verify:
                    new_w = np.copy(layer_mmio.array[:layer_w.shape[0]])
                    assert (layer_w == new_w).all()
        if flush_accel:
            # run accelerator to flush any stale weights from weight streamer FIFOs
            self.execute()

    def fold_input(self, ibuf_normal):
        \"\"\"Reshapes input in desired shape.
        Gets input data (ibuf_normal), checks if data is in expected normal shape.
        Returns folded input.\"\"\"
        # ensure that shape is as expected
        assert ibuf_normal.shape == self.ishape_normal
        # convert to folded form
        ibuf_folded = ibuf_normal.reshape(self.ishape_folded)
        return ibuf_folded

    def pack_input(self, ibuf_folded):
        \"\"\"Packs folded input and reverses both SIMD dim and endianness.
        Gets input data in folded shape and returns packed input data.\"\"\"
        ibuf_packed = finnpy_to_packed_bytearray(
            ibuf_folded, self.idt, reverse_endian=True, reverse_inner=True
        )
        return ibuf_packed

    def unpack_output(self, obuf_packed):
        \"\"\"Unpacks the packed output buffer from accelerator.
        Gets packed output and returns output data in folded shape.\"\"\"
        obuf_folded = packed_bytearray_to_finnpy(
            obuf_packed, self.odt, self.oshape_folded, reverse_endian=True, reverse_inner=True
        )
        return obuf_folded

    def unfold_output(self, obuf_folded):
        \"\"\"Unfolds output data to normal shape.
        Gets folded output data and returns output data in normal shape.\"\"\"
        obuf_normal = obuf_folded.reshape(self.oshape_normal)
        return obuf_normal

    def copy_input_data_to_device(self, data):
        \"\"\"Copies given input data to PYNQ buffer.\"\"\"
        np.copyto(self.ibuf_packed_device, data)
        self.ibuf_packed_device.flush()

    def copy_output_data_from_device(self, data):
        \"\"\"Copies PYNQ output buffer from device.\"\"\"
        self.obuf_packed_device.invalidate()
        np.copyto(data, self.obuf_packed_device)

    def execute(self):
        \"\"\"Executes accelerator by setting up the DMA(s) and
        waiting until all transfers/calls complete. Uses only member variables and
        returns nothing.\"\"\"
        if self.platform == "zynq-iodma":
            # manually launch IODMAs since signatures are missing
            self.idma.write(0x10, self.ibuf_packed_device.device_address)
            self.idma.write(0x1c, self.N)
            self.odma.write(0x10, self.obuf_packed_device.device_address)
            self.odma.write(0x1c, self.N)
            self.idma.write(0x00, 1)
            self.odma.write(0x00, 1)
            # wait until output IODMA is finished
            status = self.odma.read(0x00)
            while status & 0x2 == 0:
                status = self.odma.read(0x00)
        elif self.platform == "alveo":
            idma_handle = self.idma.start_sw(self.ibuf_packed_device, self.N)
            odma_handle = self.odma.start_sw(self.obuf_packed_device, self.N)
            odma_handle.wait()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set exec mode, batchsize N, bitfile name, inputfile name and outputfile name')
    parser.add_argument('--exec_mode', help='Please select functional verification ("execute") or throughput test ("throughput_test")', default="execute")
    parser.add_argument('--platform', help='Target platform: zynq-iodma alveo', default="$PLATFORM$")
    parser.add_argument('--batchsize', help='number of samples for inference', type=int, default=1)
    parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit")
    parser.add_argument('--inputfile', help='name of input npy file (i.e. "input.npy")', default="input.npy")
    parser.add_argument('--outputfile', help='name of output npy file (i.e. "output.npy")', default="output.npy")
    # parse arguments
    args = parser.parse_args()
    exec_mode = args.exec_mode
    platform = args.platform
    N = args.batchsize
    bitfile = args.bitfile
    inputfile = args.inputfile
    outputfile = args.outputfile

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    finnDriver = FINNAccelDriver(N, bitfile, platform)

    # for the remote execution the data from the input npy file has to be loaded,
    # packed and copied to the PYNQ buffer
    if exec_mode == "execute":
        # remove old output file to prevent reusing old output
        # in case execution fails
        try:
            os.remove(outputfile)
        except FileNotFoundError:
            pass
        # load desired input .npy file
        ibuf_normal = np.load(inputfile)
        ibuf_folded = finnDriver.fold_input(ibuf_normal)
        ibuf_packed = finnDriver.pack_input(ibuf_folded)
        finnDriver.copy_input_data_to_device(ibuf_packed)
    elif exec_mode != "throughput_test":
        raise Exception("Exec mode has to be set to remote_pynq or throughput_test")

    # for the throughput test the runtime of the network has to be measured
    if exec_mode == "throughput_test":
        # remove old metrics file
        try:
            os.remove("nw_metrics.txt")
        except FileNotFoundError:
            pass
        # dictionary for results of throughput test
        res={}
        # measure runtime of network
        start = time.time()

    # execute accelerator
    finnDriver.execute()

    # measure run time and fill dictionary with results of the throughput test
    if exec_mode == "throughput_test":
        end = time.time()
        runtime = end - start
        res["runtime[ms]"] = runtime*1000
        res["throughput[images/s]"] = N / runtime
        res["DRAM_in_bandwidth[Mb/s]"] = np.prod(finnDriver.ishape_packed)*0.000001 / runtime
        res["DRAM_out_bandwidth[Mb/s]"] = np.prod(finnDriver.oshape_packed)*0.000001 / runtime
        if platform != "alveo":
            res["fclk[mhz]"] = Clocks.fclk0_mhz
        else:
            res["fclk[mhz]"] = finnDriver.fclk_mhz
        res["N"] = N
        file = open("nw_metrics.txt", "w")
        file.write(str(res))
        file.close()

    # if execution is selected unpack, unfold and save output to output npy file
    else:
        obuf_packed = np.empty_like(finnDriver.obuf_packed_device)
        finnDriver.copy_output_data_from_device(obuf_packed)
        obuf_folded = finnDriver.unpack_output(obuf_packed)
        obuf_normal = finnDriver.unfold_output(obuf_folded)
        np.save(outputfile, obuf_normal)


"""

custom_zynq_shell_template = """
set FREQ_MHZ %s
set NUM_AXILITE %d
if {$NUM_AXILITE > 9} {
    error "Maximum 10 AXI-Lite interfaces supported"
}
set NUM_AXIMM %d
set BOARD %s
set FPGA_PART %s
create_project finn_zynq_link ./ -part $FPGA_PART

# set board part repo paths to find PYNQ-Z1/Z2
set paths_prop [get_property BOARD_PART_REPO_PATHS [current_project]]
set paths_param [get_param board.repoPaths]
lappend paths_prop /workspace/finn/board_files
lappend paths_param /workspace/finn/board_files
set_property BOARD_PART_REPO_PATHS $paths_prop [current_project]
set_param board.repoPaths $paths_param

if {$BOARD == "ZCU104"} {
    set_property board_part xilinx.com:zcu104:part0:1.1 [current_project]
    set ZYNQ_TYPE "zynq_us+"
} elseif {$BOARD == "Ultra96"} {
    set_property board_part em.avnet.com:ultra96v1:part0:1.2 [current_project]
    set ZYNQ_TYPE "zynq_us+"
} elseif {$BOARD == "Pynq-Z2"} {
    set ZYNQ_TYPE "zynq_7000"
} elseif {$BOARD == "Pynq-Z1"} {
    set ZYNQ_TYPE "zynq_7000"
    set_property board_part www.digilentinc.com:pynq-z1:part0:1.0 [current_project]
} else {
    puts "Unrecognized board"
}

create_bd_design "top"
if {$ZYNQ_TYPE == "zynq_us+"} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.3 zynq_ps
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ps]
    #activate one slave port, deactivate the second master port
    set_property -dict [list CONFIG.PSU__USE__S_AXI_GP2 {1}] [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PSU__USE__M_AXI_GP1 {0}] [get_bd_cells zynq_ps]
    #set frequency of PS clock (this can't always be exactly met)
    set_property -dict [list CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ [expr int($FREQ_MHZ)]] [get_bd_cells zynq_ps]
} elseif {$ZYNQ_TYPE == "zynq_7000"} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 zynq_ps
    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ [expr int($FREQ_MHZ)]] [get_bd_cells zynq_ps]
} else {
    puts "Unrecognized Zynq type"
}

#instantiate axi interconnect, axi smartconnect
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
#set number of axilite interfaces, and number of axi master interfaces
set_property -dict [list CONFIG.NUM_SI $NUM_AXIMM] [get_bd_cells smartconnect_0]
set_property -dict [list CONFIG.NUM_MI $NUM_AXILITE] [get_bd_cells axi_interconnect_0]

#create reset controller and connect interconnects to PS
if {$ZYNQ_TYPE == "zynq_us+"} {
    connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]
    connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] -boundary_type upper [get_bd_intf_pins axi_interconnect_0/S00_AXI]
    #connect interconnect clocks and resets
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/S00_ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins zynq_ps/saxihp0_fpd_aclk]
} elseif {$ZYNQ_TYPE == "zynq_7000"} {
    connect_bd_intf_net -boundary_type upper [get_bd_intf_pins zynq_ps/M_AXI_GP0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins zynq_ps/S_AXI_HP0]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/FCLK_CLK0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/FCLK_CLK0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/S00_ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/FCLK_CLK0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins zynq_ps/S_AXI_HP0_ACLK]
}
connect_bd_net [get_bd_pins axi_interconnect_0/ARESETN] [get_bd_pins smartconnect_0/aresetn]

#custom IP instantiations/connections start here
%s

# set up debug
if {%d == 1} {
    set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {idma0_m_axis_0}]
    set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {StreamingDataflowPartition_1_m_axis_0}]
    set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {smartconnect_0_M00_AXI}]
    apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list \
                                                              [get_bd_intf_nets smartconnect_0_M00_AXI] {AXI_R_ADDRESS "Data and Trigger" AXI_R_DATA "Data and Trigger" AXI_W_ADDRESS "Data and Trigger" AXI_W_DATA "Data and Trigger" AXI_W_RESPONSE "Data and Trigger" CLK_SRC "/zynq_ps/FCLK_CLK0" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                              [get_bd_intf_nets idma0_m_axis_0] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/zynq_ps/FCLK_CLK0" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                              [get_bd_intf_nets StreamingDataflowPartition_1_m_axis_0] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/zynq_ps/FCLK_CLK0" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                             ]
}

#finalize clock and reset connections for interconnects
if {$ZYNQ_TYPE == "zynq_us+"} {
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} }  [get_bd_pins axi_interconnect_0/M*_ACLK]
} elseif {$ZYNQ_TYPE == "zynq_7000"} {
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/FCLK_CLK0} }  [get_bd_pins axi_interconnect_0/M*_ACLK]
}

save_bd_design
assign_bd_address
validate_bd_design

set_property SYNTH_CHECKPOINT_MODE "Hierarchical" [ get_files top.bd ]
make_wrapper -files [get_files top.bd] -import -fileset sources_1 -top

set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AlternateRoutability [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property strategy Performance_ExtraTimingOpt [get_runs impl_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]

# out-of-context synth can't be used for bitstream generation
# set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-mode out_of_context} -objects [get_runs synth_1]
launch_runs -to_step write_bitstream impl_1
wait_on_run [get_runs impl_1]

# generate synthesis report
open_run impl_1
report_utilization -hierarchical -hierarchical_depth 4 -file synth_report.xml -format xml
close_project
"""

alveo_run_sh_template = """#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: alveo_run.sh <exec_mode={execute, throughput_test}> <batch_size>"
    exit -1
fi

cd $REMOTE_DEPLOY_DIR$
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME$
source $REMOTE_XRT$/setup.sh
export PLATFORM_REPO_PATHS=$REMOTE_PLATFORM_REPO_PATHS$
python3.6 driver.py --exec_mode=$1 --batchsize=$2 --bitfile=$BITFILE$ \
    --inputfile=input.npy --outputfile=output.npy --platform=alveo
"""

vitis_gen_xml_report_tcl_template = """
open_project $VITIS_PROJ_PATH$/_x/link/vivado/vpl/prj/prj.xpr
open_run impl_1
report_utilization -hierarchical -hierarchical_depth 5 -file $VITIS_PROJ_PATH$/synth_report.xml -format xml
"""

pynq_validation_template = """
import argparse
from driver import FINNAccelDriver
import numpy as np

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Validate top-1 accuracy for FINN accelerator')
  parser.add_argument('--batchsize', help='number of samples for inference', type=int, default=100)
  parser.add_argument('--dataset', help='dataset to use (mnist of cifar10)', required=True)
  # parse arguments
  args = parser.parse_args()
  bsize = args.batchsize
  dataset = args.dataset

  if dataset == "mnist":
    from dataset_loading import mnist
    trainx, trainy, testx, testy, valx, valy = mnist.load_mnist_data("/tmp", download=True, one_hot=False)
  elif dataset == "cifar10":
    from dataset_loading import cifar
    trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data("/tmp", download=True, one_hot=False)
  else:
    raise Exception("Unrecognized dataset")

  test_imgs = testx
  test_labels = testy

  ok = 0
  nok = 0
  total = test_imgs.shape[0]
  driver = FINNAccelDriver(bsize, "resizer.bit", "zynq-iodma")

  n_batches = int(total / bsize)

  test_imgs = test_imgs.reshape(n_batches, bsize, -1)
  test_labels = test_labels.reshape(n_batches, bsize)

  for i in range(n_batches):
    ibuf_normal = test_imgs[i].reshape(driver.ibuf_packed_device.shape)
    exp = test_labels[i]
    driver.copy_input_data_to_device(ibuf_normal)
    driver.execute()
    obuf_normal = np.empty_like(driver.obuf_packed_device)
    driver.copy_output_data_from_device(obuf_normal)
    ret = np.bincount(obuf_normal.flatten() == exp.flatten())
    nok += ret[0]
    ok += ret[1]
    print("batch %d / %d : total OK %d NOK %d" % (i, n_batches, ok, nok))

  acc = 100.0 * ok / (total)
  print("Final accuracy: %f" % acc)
"""
