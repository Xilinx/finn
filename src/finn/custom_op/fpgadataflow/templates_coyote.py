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

# template for single node execution
from typing import Dict, List

import finn.transformation.fpgadataflow.coyote_build as coyote_build

hls_bridge = """
void write_intf_bridge(unsigned value_finn, unsigned *ptr){
#pragma HLS INTERFACE m_axi port=ptr offset=slave
#pragma HLS INTERFACE s_axilite port=value_finn
#pragma HLS INTERFACE s_axilite port=return
    *ptr = value_finn;
}
"""

hls_bridge_ip_name = "hls_bridge"
hls_bridge_version = "1.0"
hls_bridge_vendor = "xilinx_finn"
hls_bridge_library = "finn"

hls_bridge_script = """
open_project hls_bridge
open_solution "hls_bridge_sol" -flow_target vivado
add_files bridge.c
set_top write_intf_bridge
set_part {$PART$}
create_clock -period 10 -name default
config_interface -m_axi_addr64=0
csynth_design
export_design -format ip_catalog -ipname "%s" -version "%s" -vendor "%s" -library "%s"
exit 0
""" % (
    hls_bridge_ip_name,
    hls_bridge_version,
    hls_bridge_vendor,
    hls_bridge_library,
)

accl_bd_intra_connections: str = """
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_dma0_mm2s_cmd] \\
    [get_bd_intf_pins cyt_dma_adapter_0/dma0_mm2s_cmd]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_dma1_mm2s_cmd] \\
    [get_bd_intf_pins cyt_dma_adapter_0/dma1_mm2s_cmd]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_dma1_s2mm_cmd] \\
    [get_bd_intf_pins cyt_dma_adapter_0/dma1_s2mm_cmd]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_dma0_s2mm_cmd] \\
    [get_bd_intf_pins cyt_dma_adapter_0/dma0_s2mm_cmd]
connect_bd_intf_net [get_bd_intf_pins cyt_dma_adapter_0/dma0_s2mm_sts] \\
    [get_bd_intf_pins ccl_offload_0/s_axis_dma0_s2mm_sts]
connect_bd_intf_net [get_bd_intf_pins cyt_dma_adapter_0/dma1_s2mm_sts] \\
    [get_bd_intf_pins ccl_offload_0/s_axis_dma1_s2mm_sts]
connect_bd_intf_net [get_bd_intf_pins cyt_dma_adapter_0/dma0_mm2s_sts] \\
    [get_bd_intf_pins ccl_offload_0/s_axis_dma0_mm2s_sts]
connect_bd_intf_net [get_bd_intf_pins cyt_dma_adapter_0/dma1_mm2s_sts] \\
    [get_bd_intf_pins ccl_offload_0/s_axis_dma1_mm2s_sts]
make_bd_pins_external  [get_bd_pins ccl_offload_0/ap_clk]
make_bd_pins_external  [get_bd_pins ccl_offload_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins cyt_dma_adapter_0/ap_clk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins cyt_dma_adapter_0/ap_rst_n]
make_bd_intf_pins_external  [get_bd_intf_pins cyt_dma_adapter_0/cyt_byp_wr_sts]
make_bd_intf_pins_external  [get_bd_intf_pins cyt_dma_adapter_0/cyt_byp_rd_sts]
make_bd_intf_pins_external  [get_bd_intf_pins cyt_dma_adapter_0/cyt_byp_wr_cmd]
make_bd_intf_pins_external  [get_bd_intf_pins cyt_dma_adapter_0/cyt_byp_rd_cmd]

connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_arith_op0] \\
    [get_bd_intf_pins reduce_ops_0/in0]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_arith_op1] \\
    [get_bd_intf_pins reduce_ops_0/in1]
connect_bd_intf_net [get_bd_intf_pins reduce_ops_0/out_r] \\
    [get_bd_intf_pins ccl_offload_0/s_axis_arith_res]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins reduce_ops_0/ap_clk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins reduce_ops_0/ap_rst_n]

connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins hostctrl_0/ap_clk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins hostctrl_0/ap_rst_n]

connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins client_arbiter_0/ap_clk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins client_arbiter_0/ap_rst_n]

connect_bd_intf_net [get_bd_intf_pins hostctrl_0/cmd] \\
    [get_bd_intf_pins client_arbiter_0/cmd_clients_0    ]
connect_bd_intf_net [get_bd_intf_pins client_arbiter_0/ack_clients_0] \\
    [get_bd_intf_pins hostctrl_0/sts    ]
make_bd_intf_pins_external [get_bd_intf_pins client_arbiter_0/cmd_clients_1]
make_bd_intf_pins_external [get_bd_intf_pins client_arbiter_0/ack_clients_1]
connect_bd_intf_net [get_bd_intf_pins client_arbiter_0/cmd_cclo] \\
    [get_bd_intf_pins ccl_offload_0/s_axis_call_req]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_call_ack] \\
    [get_bd_intf_pins client_arbiter_0/ack_cclo]

make_bd_intf_pins_external [get_bd_intf_pins ccl_offload_0/m_axis_krnl]
make_bd_intf_pins_external [get_bd_intf_pins ccl_offload_0/s_axis_krnl]

connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_compression0] \\
    [get_bd_intf_pins ccl_offload_0/s_axis_compression0]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_compression1] \\
    [get_bd_intf_pins ccl_offload_0/s_axis_compression1]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_compression2] \\
    [get_bd_intf_pins ccl_offload_0/s_axis_compression2]

# externalize RDMA streams
# data streams
set m_axis_eth_tx_data \\
    [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_eth_tx_data ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_eth_tx_data
set s_axis_eth_rx_data \\
    [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_eth_rx_data ]
set_property -dict \\
    [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} \\
        CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {1} CONFIG.LAYERED_METADATA {undef} \\
        CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {8} CONFIG.TID_WIDTH {0} \\
        CONFIG.TUSER_WIDTH {0} ] $s_axis_eth_rx_data

# RDMA sq and rq
set m_axis_rdma_sq \\
    [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_rdma_sq ]
set_property -dict \\
    [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_rdma_sq

# RDMA extra pair of host/card streams
set m_axis_host_2 \\
    [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_host_2 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_host_2
set m_axis_card_2 \\
    [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_card_2 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_card_2
set s_axis_host_2 \\
    [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_host_2 ]
set_property -dict \\
    [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} \\
        CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} \\
        CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} \\
        CONFIG.TUSER_WIDTH {0} ] $s_axis_host_2
set s_axis_card_2 \\
    [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_card_2 ]
set_property -dict \\
    [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} \\
    CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} \\
    CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} \\
    CONFIG.TUSER_WIDTH {0} ] $s_axis_card_2

# RDMA wr_req and rd_req
set s_axis_rdma_wr_req \\
    [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_rdma_wr_req ]
set_property -dict \\
    [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} \\
        CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} \\
        CONFIG.TDATA_NUM_BYTES {12} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} \\
        CONFIG.TUSER_WIDTH {0} ] $s_axis_rdma_wr_req
set s_axis_rdma_rd_req \\
    [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_rdma_rd_req ]
set_property -dict \\
    [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} \\
        CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} \\
        CONFIG.TDATA_NUM_BYTES {12} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} \\
            CONFIG.TUSER_WIDTH {0} ] $s_axis_rdma_rd_req

# connections for rdma_arbiter and the axi 1-to-2 switch
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins cyt_rdma_arbiter_0/ap_clk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins cyt_rdma_arbiter_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins axis_switch_1_to_2_inst_2/aclk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins axis_switch_1_to_2_inst_2/aresetn]

connect_bd_intf_net \\
    [get_bd_intf_ports s_axis_eth_rx_data] [get_bd_intf_pins cyt_rdma_arbiter_0/s_axis]
connect_bd_intf_net \\
    [get_bd_intf_ports s_axis_rdma_wr_req] [get_bd_intf_pins cyt_rdma_arbiter_0/s_meta]
connect_bd_intf_net \\
    [get_bd_intf_pins cyt_rdma_arbiter_0/m_meta_0] \\
        [get_bd_intf_pins ccl_offload_0/s_axis_eth_notification]
connect_bd_intf_net \\
    [get_bd_intf_pins cyt_rdma_arbiter_0/m_axis_0] \\
        [get_bd_intf_pins ccl_offload_0/s_axis_eth_rx_data]
connect_bd_intf_net \\
    [get_bd_intf_pins cyt_rdma_arbiter_0/m_meta_1] [get_bd_intf_pins cyt_dma_adapter_0/rdma_wr_req]
connect_bd_intf_net \\
    [get_bd_intf_pins cyt_rdma_arbiter_0/m_axis_1] \\
        [get_bd_intf_pins axis_switch_1_to_2_inst_2/S00_AXIS]
connect_bd_intf_net \\
    [get_bd_intf_pins axis_switch_1_to_2_inst_2/M00_AXIS] [get_bd_intf_ports m_axis_card_2]
connect_bd_intf_net \\
    [get_bd_intf_pins axis_switch_1_to_2_inst_2/M01_AXIS] [get_bd_intf_ports m_axis_host_2]

# connections for rdma_mux and the axi 2-to-1 switch
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins cyt_rdma_mux_0/ap_clk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins cyt_rdma_mux_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins axis_switch_2_to_1_inst_2/aclk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins axis_switch_2_to_1_inst_2/aresetn]
connect_bd_net \\
    [get_bd_pins xlconstant_2/dout] [get_bd_pins axis_switch_2_to_1_inst_2/s_req_suppress]

connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/s_meta_0] \\
    [get_bd_intf_pins ccl_offload_0/m_axis_rdma_sq]
connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/s_axis_0] \\
    [get_bd_intf_pins ccl_offload_0/m_axis_eth_tx_data]
connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/s_meta_1] \\
    [get_bd_intf_ports s_axis_rdma_rd_req]
connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/s_axis_1] \\
    [get_bd_intf_pins axis_switch_2_to_1_inst_2/M00_AXIS]
connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/m_meta_0] \\
    [get_bd_intf_ports m_axis_rdma_sq]
connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/m_meta_1] \\
    [get_bd_intf_pins cyt_dma_adapter_0/rdma_rd_req]
connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/m_axis] [get_bd_intf_ports m_axis_eth_tx_data]
connect_bd_intf_net [get_bd_intf_ports s_axis_host_2] \\
    [get_bd_intf_pins axis_switch_2_to_1_inst_2/S00_AXIS]
connect_bd_intf_net [get_bd_intf_ports s_axis_card_2] \\
    [get_bd_intf_pins axis_switch_2_to_1_inst_2/S01_AXIS]

# externalize DMA data streams

set m_axis_host_0 \\
    [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_host_0 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_host_0
set m_axis_host_1 \\
    [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_host_1 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_host_1
set m_axis_card_0 \\
    [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_card_0 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_card_0
set m_axis_card_1 \\
    [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_card_1 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_card_1

set s_axis_host_0 \\
    [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_host_0 ]
set_property -dict \\
    [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} \\
        CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} \\
        CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} \\
        CONFIG.TUSER_WIDTH {0} ] $s_axis_host_0
set s_axis_host_1 \\
    [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_host_1 ]
set_property -dict \\
    [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} \\
        CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} \\
        CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} \\
        CONFIG.TUSER_WIDTH {0} ] $s_axis_host_1
set s_axis_card_0 \\
    [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_card_0 ]
set_property -dict \\
    [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} \\
        CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} \\
        CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} \\
        CONFIG.TUSER_WIDTH {0} ] $s_axis_card_0
set s_axis_card_1 \\
    [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_card_1 ]
set_property -dict \\
    [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} \\
        CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} \\
            CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} \\
                CONFIG.TUSER_WIDTH {0} ] $s_axis_card_1


# s_axis_host_0 and s_axis_card_0 multiplexed to single s_axis_dma0_mm2s stream
# round-robin by tlast
connect_bd_intf_net \\
    [get_bd_intf_ports s_axis_host_0] [get_bd_intf_pins axis_switch_2_to_1_inst_0/S00_AXIS]
connect_bd_intf_net \\
    [get_bd_intf_ports s_axis_card_0] [get_bd_intf_pins axis_switch_2_to_1_inst_0/S01_AXIS]
connect_bd_intf_net \\
    [get_bd_intf_pins axis_switch_2_to_1_inst_0/M00_AXIS] \\
    [get_bd_intf_pins ccl_offload_0/s_axis_dma0_mm2s]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins axis_switch_2_to_1_inst_0/aclk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins axis_switch_2_to_1_inst_0/aresetn]

connect_bd_net [get_bd_pins xlconstant_0/dout] \\
    [get_bd_pins axis_switch_2_to_1_inst_0/s_req_suppress]

# s_axis_host_1 and s_axis_card_1 multiplexed to single s_axis_dma1_mm2s stream
# round-robin by tlast
connect_bd_intf_net [get_bd_intf_ports s_axis_host_1] \\
    [get_bd_intf_pins axis_switch_2_to_1_inst_1/S00_AXIS]
connect_bd_intf_net [get_bd_intf_ports s_axis_card_1] \\
    [get_bd_intf_pins axis_switch_2_to_1_inst_1/S01_AXIS]
connect_bd_intf_net [get_bd_intf_pins axis_switch_2_to_1_inst_1/M00_AXIS] \\
    [get_bd_intf_pins ccl_offload_0/s_axis_dma1_mm2s]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins axis_switch_2_to_1_inst_1/aclk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins axis_switch_2_to_1_inst_1/aresetn]

connect_bd_net [get_bd_pins xlconstant_1/dout] \\
    [get_bd_pins axis_switch_2_to_1_inst_1/s_req_suppress]

# m_axis_dma0_s2mm multiplex to m_axis_host_0 and m_axis_card_0 according to
# the strm flag encoded in m_axis_dma0_s2mm tdest
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_dma0_s2mm] \\
    [get_bd_intf_pins axis_switch_1_to_2_inst_0/S00_AXIS]
connect_bd_intf_net [get_bd_intf_ports m_axis_card_0] \\
    [get_bd_intf_pins axis_switch_1_to_2_inst_0/M00_AXIS]
connect_bd_intf_net [get_bd_intf_ports m_axis_host_0] \\
    [get_bd_intf_pins axis_switch_1_to_2_inst_0/M01_AXIS]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins axis_switch_1_to_2_inst_0/aclk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins axis_switch_1_to_2_inst_0/aresetn]

# m_axis_dma1_s2mm multiplex to m_axis_host_1 and m_axis_card_1 according to
# the strm flag encoded in m_axis_dma1_s2mm tdest
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_dma1_s2mm] \\
    [get_bd_intf_pins axis_switch_1_to_2_inst_1/S00_AXIS]
connect_bd_intf_net [get_bd_intf_ports m_axis_card_1] \\
    [get_bd_intf_pins axis_switch_1_to_2_inst_1/M00_AXIS]
connect_bd_intf_net [get_bd_intf_ports m_axis_host_1] \\
    [get_bd_intf_pins axis_switch_1_to_2_inst_1/M01_AXIS]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins axis_switch_1_to_2_inst_1/aclk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins axis_switch_1_to_2_inst_1/aresetn]



# connect up AXI lite
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins smartconnect_0/aresetn]
connect_bd_intf_net [get_bd_intf_pins hostctrl_0/s_axi_control] \\
    [get_bd_intf_pins smartconnect_0/M00_AXI]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/s_axi_control] \\
    [get_bd_intf_pins smartconnect_0/M01_AXI]
make_bd_intf_pins_external  [get_bd_intf_pins smartconnect_0/S00_AXI]
set_property -dict [list CONFIG.ADDR_WIDTH {16}] [get_bd_intf_ports S00_AXI_0]

# Create address segments
assign_bd_address -offset 0x00000000 -range 0x00002000 -target_address_space \\
    [get_bd_addr_spaces S00_AXI_0] [get_bd_addr_segs ccl_offload_0/s_axi_control/reg0] -force
assign_bd_address -offset 0x00002000 -range 0x00002000 -target_address_space \\
    [get_bd_addr_spaces S00_AXI_0] [get_bd_addr_segs hostctrl_0/s_axi_control/Reg] -force

set_property CONFIG.PROTOCOL AXI4LITE [get_bd_intf_ports /S00_AXI_0]
set_property -dict [list CONFIG.HAS_BURST {0} CONFIG.HAS_CACHE {0} CONFIG.HAS_LOCK {0} \\
    CONFIG.HAS_QOS {0} CONFIG.HAS_REGION {0}] [get_bd_intf_ports S00_AXI_0]
"""

user_logic_config = """
// Constants
localparam integer COYOTE_AXIL_ADDR_LSB = $clog2(AXIL_DATA_BITS/8);
localparam integer COYOTE_AXIL_ADDR_MSB = 16;

// Master Data Stream
AXI4SR axis_host_0_src_s ();
AXI4SR axis_host_1_src_s ();
AXI4SR axis_host_2_src_s ();
AXI4SR axis_card_0_src_s ();
AXI4SR axis_card_1_src_s ();
AXI4SR axis_card_2_src_s ();

// register slices
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_0_src_s),
  .m_axis(axis_host_0_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_1_src_s),
  .m_axis(axis_host_1_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_2_src_s),
  .m_axis(axis_host_2_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_0_src_s),
  .m_axis(axis_card_0_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_1_src_s),
  .m_axis(axis_card_1_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_2_src_s),
  .m_axis(axis_card_2_src));

// Slave Data Stream
AXI4SR axis_host_0_sink_s ();
AXI4SR axis_host_1_sink_s ();
AXI4SR axis_host_2_sink_s ();
AXI4SR axis_card_0_sink_s ();
AXI4SR axis_card_1_sink_s ();
AXI4SR axis_card_2_sink_s ();

// register slices
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_0_sink),
  .m_axis(axis_host_0_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_1_sink),
  .m_axis(axis_host_1_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_2_sink),
  .m_axis(axis_host_2_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_0_sink),
  .m_axis(axis_card_0_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_1_sink),
  .m_axis(axis_card_1_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_2_sink),
  .m_axis(axis_card_2_sink_s));

assign axis_host_0_src_s.tid = 0;
assign axis_host_1_src_s.tid = 0;
assign axis_host_2_src_s.tid = 0;

assign axis_card_0_src_s.tid = 0;
assign axis_card_1_src_s.tid = 0;
assign axis_card_2_src_s.tid = 0;

assign rdma_0_ack.ready = 1'b1;

"""


def generate_accl_bd(accl_kernel_path: str):
    config_axis_switch_2_to_1: Dict[str, str] = {
        "NUM_SI": "2",
        "TDATA_NUM_BYTES": "64",
        "HAS_TKEEP": "1",
        "HAS_TLAST": "1",
        "ARB_ON_TLAST": "1",
        "NUM_MI": "1",
        "DECODER_REG": "0",
    }
    config_axis_switch_1_to_2: Dict[str, str] = {
        "NUM_SI": "1",
        "NUM_MI": "2",
        "TDATA_NUM_BYTES": "64",
        "HAS_TKEEP": "1",
        "HAS_TLAST": "1",
        "TDEST_WIDTH": "8",
        "DECODER_REG": "1",
    }
    ips: List[coyote_build.IP] = [
        coyote_build.IP(
            vlnv="Xilinx:ACCL:ccl_offload:1.0",
            module_name="ccl_offload_0",
            interfaces=[],
            ip_repo_path=accl_kernel_path,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ACCL:cyt_dma_adapter:1.0",
            module_name="cyt_dma_adapter_0",
            interfaces=[],
            ip_repo_path=accl_kernel_path,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ACCL:reduce_ops:1.0",
            module_name="reduce_ops_0",
            interfaces=[],
            ip_repo_path=accl_kernel_path,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ACCL:hostctrl:1.0",
            module_name="hostctrl_0",
            interfaces=[],
            ip_repo_path=accl_kernel_path,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ACCL:client_arbiter:1.0",
            module_name="client_arbiter_0",
            interfaces=[],
            ip_repo_path=accl_kernel_path,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ip:axis_switch:1.1",
            module_name="axis_switch_2_to_1_inst_0",
            interfaces=[],
            ip_repo_path=None,
            config=config_axis_switch_2_to_1,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ip:axis_switch:1.1",
            module_name="axis_switch_2_to_1_inst_1",
            interfaces=[],
            ip_repo_path=None,
            config=config_axis_switch_2_to_1,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ip:axis_switch:1.1",
            module_name="axis_switch_1_to_2_inst_0",
            interfaces=[],
            ip_repo_path=None,
            config=config_axis_switch_1_to_2,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ip:axis_switch:1.1",
            module_name="axis_switch_1_to_2_inst_1",
            interfaces=[],
            ip_repo_path=None,
            config=config_axis_switch_1_to_2,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ACCL:cyt_rdma_arbiter:1.0",
            module_name="cyt_rdma_arbiter_0",
            interfaces=[],
            ip_repo_path=accl_kernel_path,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ip:axis_switch:1.1",
            module_name="axis_switch_1_to_2_inst_2",
            interfaces=[],
            ip_repo_path=None,
            config=config_axis_switch_1_to_2,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ACCL:cyt_rdma_mux:1.0",
            module_name="cyt_rdma_mux_0",
            interfaces=[],
            ip_repo_path=accl_kernel_path,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ip:axis_switch:1.1",
            module_name="axis_switch_2_to_1_inst_2",
            interfaces=[],
            ip_repo_path=None,
            config=config_axis_switch_2_to_1,
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ip:xlconstant:1.1",
            module_name="xlconstant_2",
            interfaces=[],
            ip_repo_path=None,
            config={
                "CONST_WIDTH": "2",
                "CONST_VAL": "0",
            },
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ip:xlconstant:1.1",
            module_name="xlconstant_0",
            interfaces=[],
            ip_repo_path=None,
            config={
                "CONST_WIDTH": "2",
                "CONST_VAL": "0",
            },
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ip:xlconstant:1.1",
            module_name="xlconstant_1",
            interfaces=[],
            ip_repo_path=None,
            config={
                "CONST_WIDTH": "2",
                "CONST_VAL": "0",
            },
        ),
        coyote_build.IP(
            vlnv="xilinx.com:ip:smartconnect:1.0",
            module_name="smartconnect_0",
            interfaces=[],
            ip_repo_path=None,
            config={"NUM_MI": "2", "NUM_SI": "1"},
        ),
    ]

    return coyote_build.BD(
        bd_name="accl_bd",
        ips=ips,
        interfaces=[
            coyote_build.Clk("ap_clk_0", 1),
            coyote_build.Reset("ap_rst_n_0", 1),
            coyote_build.AXI4Lite(
                name="S00_AXI_0",
                width=32,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                addr_width=16,
                shift_left=1,
            ),
            coyote_build.AXI4Stream(
                name="ack_clients_1_0",
                data_width=32,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                tstrb=True,
            ),
            coyote_build.AXI4Stream(
                name="cmd_clients_1_0",
                data_width=32,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                tstrb=True,
            ),
            coyote_build.AXI4Stream(
                name="cyt_byp_rd_cmd_0",
                data_width=96,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            coyote_build.AXI4Stream(
                name="cyt_byp_rd_sts_0",
                data_width=16,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            coyote_build.AXI4Stream(
                name="cyt_byp_wr_cmd_0",
                data_width=96,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            coyote_build.AXI4Stream(
                name="cyt_byp_wr_sts_0",
                data_width=16,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            coyote_build.AXI4Stream(
                name="m_axis_card_0",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                # tdest=8,
            ),
            coyote_build.AXI4Stream(
                name="m_axis_card_1",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                # tdest=8,
            ),
            coyote_build.AXI4Stream(
                name="m_axis_card_2",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                # tdest=8,
                # tstrb=True,
            ),
            # TODO: tdest connected to tid
            coyote_build.AXI4Stream(
                name="m_axis_eth_tx_data",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                tdest=8,
                connect_tid_to_tdest=True,
                # tstrb=True,
            ),
            coyote_build.AXI4Stream(
                name="m_axis_host_0",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                # tdest=8,
            ),
            coyote_build.AXI4Stream(
                name="m_axis_host_1",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                # tdest=8,
            ),
            coyote_build.AXI4Stream(
                name="m_axis_host_2",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                # tdest=8,
                # tstrb=True,
            ),
            coyote_build.AXI4Stream(
                name="m_axis_krnl_0",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                # tdest=8,
            ),
            coyote_build.AXI4Stream(
                name="m_axis_rdma_sq",
                data_width=512,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            coyote_build.AXI4Stream(
                name="s_axis_card_0",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            coyote_build.AXI4Stream(
                name="s_axis_card_1",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            coyote_build.AXI4Stream(
                name="s_axis_card_2",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            # TODO: tdest connected to tid
            coyote_build.AXI4Stream(
                name="s_axis_eth_rx_data",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                tdest=8,
                connect_tid_to_tdest=True,
                # tstrb=True,
            ),
            coyote_build.AXI4Stream(
                name="s_axis_host_0",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            coyote_build.AXI4Stream(
                name="s_axis_host_1",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            coyote_build.AXI4Stream(
                name="s_axis_host_2",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            coyote_build.AXI4Stream(
                name="s_axis_krnl_0",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            coyote_build.AXI4Stream(
                name="s_axis_rdma_rd_req",
                data_width=96,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
            coyote_build.AXI4Stream(
                name="s_axis_rdma_wr_req",
                data_width=96,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
            ),
        ],
        intra_connections=accl_bd_intra_connections.splitlines(),
        extra_external_commands=[],
        # run_needed=False,
    )


def get_coyote_interface():
    return coyote_build.ExternalInterface(
        interfaces=[
            coyote_build.AXI4Lite(
                name="axi_ctrl",
                width=64,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                addr_width=64,
            ),
            coyote_build.Clk(name="aclk", width=1),
            coyote_build.Reset(name="aresetn", width=1),
            coyote_build.AXI4Stream(
                name="axis_host_0_sink",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_host_0_src",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
            ),
        ]
    )


def get_coyote_interface_accl():
    return coyote_build.ExternalInterface(
        interfaces=[
            coyote_build.AXI4Lite(
                name="axi_ctrl",
                width=64,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                addr_width=64,
            ),
            coyote_build.Clk(name="aclk", width=1),
            coyote_build.Reset(name="aresetn", width=1),
            coyote_build.AXI4Stream(
                name="axis_host_0_sink_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
            ),
            # NOTE: No tdest for src as ACCL signals are not connected to it
            coyote_build.AXI4Stream(
                name="axis_host_0_src_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_host_0_sink_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_host_1_src_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_host_1_sink_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_host_2_src_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                # tstrb=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_host_2_sink_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_host_3_src",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                # tstrb=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_host_3_sink",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_card_0_src_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                # tdest=8,
            ),
            coyote_build.AXI4Stream(
                name="axis_card_0_sink_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_card_1_src_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                # tdest=8,
            ),
            coyote_build.AXI4Stream(
                name="axis_card_1_sink_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_card_2_src_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                # tdest=8,
                # tstrb=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_card_2_sink_s",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
            ),
            # TODO: Handle connection from tid to tdest here
            coyote_build.AXI4Stream(
                name="axis_rdma_0_sink",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                tid=8,
                connect_tid_to_tdest=True,
                # tstrb=True,
            ),
            coyote_build.AXI4Stream(
                name="axis_rdma_0_src",
                data_width=512,
                tlast=True,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                tid=8,
                connect_tid_to_tdest=True,
                # tstrb=True,
            ),
            coyote_build.AXI4Stream(
                name="bpss_rd_req",
                data_width=96,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                is_meta_intf=True,
            ),
            coyote_build.AXI4Stream(
                name="bpss_wr_req",
                data_width=96,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                is_meta_intf=True,
            ),
            coyote_build.AXI4Stream(
                name="bpss_rd_done",
                data_width=16,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                is_meta_intf=True,
            ),
            coyote_build.AXI4Stream(
                name="bpss_wr_done",
                data_width=16,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                is_meta_intf=True,
            ),
            coyote_build.AXI4Stream(
                name="rdma_0_rd_req",
                data_width=96,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                is_meta_intf=True,
            ),
            coyote_build.AXI4Stream(
                name="rdma_0_wr_req",
                data_width=96,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                is_meta_intf=True,
            ),
            coyote_build.AXI4Stream(
                name="rdma_0_sq",
                data_width=512,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                is_meta_intf=True,
            ),
            # NOTE: This is not used in the design
            coyote_build.AXI4Stream(
                name="rdma_0_ack",
                data_width=16,
                tlast=False,
                delimiter=coyote_build.AXIInterface.Delimiter.POINT,
                external=True,
                is_meta_intf=True,
            ),
        ]
    )


def get_hls_bridge_ip(path_to_hls_bridge_ip: str):
    return coyote_build.IP(
        vlnv=coyote_build.IP.build_vlnv(
            hls_bridge_vendor,
            hls_bridge_library,
            hls_bridge_ip_name,
            hls_bridge_version,
        ),
        module_name="hls_bridge_0",
        interfaces=[
            coyote_build.AXI4Lite(
                name="s_axi_control",
                width=32,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                addr_width=5,
                upper_case=True,
            ),
            coyote_build.AXI4Lite(
                name="m_axi_gmem",
                width=32,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                addr_width=32,
                upper_case=True,
            ),
            coyote_build.Clk("ap_clk", 1),
            coyote_build.Reset("ap_rst_n", 1),
        ],
        ip_repo_path=path_to_hls_bridge_ip,
        config=None,
        # run_needed=True,
    )


def get_finn_interface(accl_mode, axilites, intf_names, model):
    finn_interfaces: List[coyote_build.Interface] = []

    for axilite, width in axilites:
        finn_interfaces.append(
            coyote_build.AXI4Lite(
                name=axilite,
                width=32,
                delimiter=coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                addr_width=width,
            )
        )

    finn_interfaces.extend(
        [
            coyote_build.Clk(intf_names["clk"][0], 1),
            coyote_build.Reset(intf_names["rst"][0], 1),
            # NOTE: Input of FINN does not have TLast
            coyote_build.AXI4Stream(
                intf_names["s_axis"][0][0],
                intf_names["s_axis"][0][1],
                False,
                coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                False,
            ),
            coyote_build.AXI4Stream(
                intf_names["m_axis"][0][0],
                intf_names["m_axis"][0][1],
                accl_mode == coyote_build.CoyoteBuild.ACCLMode.ACCL_TLAST
                or accl_mode == coyote_build.CoyoteBuild.ACCLMode.NONE,
                coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                False,
            ),
        ]
    )

    if accl_mode != coyote_build.CoyoteBuild.ACCLMode.NONE:
        finn_interfaces.extend(
            [
                coyote_build.AXI4Stream(
                    intf_names["s_axis"][1][0],
                    intf_names["s_axis"][1][1],
                    False,
                    coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                    False,
                ),
                coyote_build.AXI4Stream(
                    intf_names["s_axis"][2][0],
                    intf_names["s_axis"][2][1],
                    False,
                    coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                    False,
                ),
                coyote_build.AXI4Stream(
                    intf_names["m_axis"][1][0],
                    intf_names["m_axis"][1][1],
                    False,
                    coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                    False,
                ),
                coyote_build.AXI4Stream(
                    intf_names["m_axis"][2][0],
                    intf_names["m_axis"][2][1],
                    False,
                    coyote_build.AXIInterface.Delimiter.UNDERSCORE,
                    False,
                ),
            ]
        )

    return coyote_build.IP(
        vlnv=model.get_metadata_prop("vivado_stitch_vlnv"),  # type: ignore
        module_name="finn_kernel_0",
        interfaces=finn_interfaces,
        ip_repo_path=model.get_metadata_prop("vivado_stitch_proj") + "/ip",  # type: ignore
    )
