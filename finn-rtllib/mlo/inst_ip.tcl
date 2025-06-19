# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# This file is subject to the Xilinx Design License Agreement located
# in the LICENSE.md file in the root directory of this repository.
#
# This file contains confidential and proprietary information of Xilinx, Inc.
# and is protected under U.S. and international copyright and other
# intellectual property laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any rights to the materials
# distributed herewith. Except as otherwise provided in a valid license issued to
# you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
# MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
# DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
# INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
# FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
# in contract or tort, including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature related to, arising
# under or in connection with these materials, including for any direct, or any
# indirect, special, incidental, or consequential loss or damage (including loss
# of data, profits, goodwill, or any type of loss or damage suffered as a result
# of any action brought by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the possibility of the
# same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-safe, or for use in
# any application requiring failsafe performance, such as life-support or safety
# devices or systems, Class III medical devices, nuclear facilities, applications
# related to the deployment of airbags, or any other applications that could lead
# to death, personal injury, or severe property or environmental damage
# (individually and collectively, "Critical Applications"). Customer assumes the
# sole risk and liability of any use of Xilinx products in Critical Applications,
# subject only to applicable laws and regulations governing limitations on product
# liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.
# CS

#
# Datamovers
#

create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1 -module_name axi_datamover_hbm_0
set_property -dict [list \
  CONFIG.c_addr_width {64} \
  CONFIG.c_include_mm2s_dre {true} \
  CONFIG.c_include_s2mm_dre {true} \
  CONFIG.c_m_axi_mm2s_data_width {256} \
  CONFIG.c_m_axi_mm2s_id_width {2} \
  CONFIG.c_m_axi_s2mm_data_width {256} \
  CONFIG.c_m_axi_s2mm_id_width {2} \
  CONFIG.c_m_axis_mm2s_tdata_width {256} \
  CONFIG.c_mm2s_burst_size {16} \
  CONFIG.c_s2mm_burst_size {16} \
  CONFIG.c_s_axis_s2mm_tdata_width {256} \
] [get_ips axi_datamover_hbm_0]

create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1 -module_name cdma_datamover
set_property -dict [list \
  CONFIG.c_include_mm2s_dre {true} \
  CONFIG.c_include_s2mm_dre {true} \
  CONFIG.c_addr_width {64} \
  CONFIG.c_m_axi_mm2s_data_width {256} \
  CONFIG.c_m_axis_mm2s_tdata_width {256} \
  CONFIG.c_mm2s_burst_size {64} \
  CONFIG.c_m_axi_s2mm_data_width {256} \
  CONFIG.c_s2mm_burst_size {64} \
  CONFIG.c_s_axis_s2mm_tdata_width {256} \
] [get_ips cdma_datamover]

create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1 -module_name cdma_datamover_rd
set_property -dict [list \
  CONFIG.c_addr_width {64} \
  CONFIG.c_enable_s2mm {0} \
  CONFIG.c_include_mm2s_dre {true} \
  CONFIG.c_m_axi_mm2s_data_width {256} \
  CONFIG.c_m_axis_mm2s_tdata_width {256} \
  CONFIG.c_mm2s_burst_size {64} \
] [get_ips cdma_datamover_rd]

create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1 -module_name cdma_datamover_wr
set_property -dict [list \
  CONFIG.c_addr_width {64} \
  CONFIG.c_enable_mm2s {0} \
  CONFIG.c_include_s2mm_dre {true} \
  CONFIG.c_m_axi_s2mm_data_width {256} \
  CONFIG.c_s2mm_burst_size {64} \
  CONFIG.c_s_axis_s2mm_tdata_width {256} \
] [get_ips cdma_datamover_wr]

#
# VIO
#

create_ip -name axis_vio -vendor xilinx.com -library ip -version 1.0 -module_name vio_top
set_property -dict [list \
  CONFIG.C_NUM_PROBE_IN {6} \
  CONFIG.C_NUM_PROBE_OUT {6} \
  CONFIG.C_PROBE_IN2_WIDTH {64} \
  CONFIG.C_PROBE_IN3_WIDTH {64} \
  CONFIG.C_PROBE_IN4_WIDTH {16} \
  CONFIG.C_PROBE_OUT1_WIDTH {32} \
  CONFIG.C_PROBE_OUT2_WIDTH {16} \
  CONFIG.C_PROBE_OUT3_WIDTH {32} \
  CONFIG.C_PROBE_OUT4_WIDTH {16} \
  CONFIG.C_PROBE_OUT5_WIDTH {32} \
  CONFIG.C_PROBE_OUT5_INIT_VAL {0x0d15ea5e} \
] [get_ips vio_top]

#
# DWCs (templates for these, must be some better way ...)
#

create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name dwc_strm_sink
set_property -dict [list \
  CONFIG.HAS_TKEEP {1} \
  CONFIG.HAS_TLAST {1} \
  CONFIG.M_TDATA_NUM_BYTES {${ILEN_BYTES}} \
  CONFIG.S_TDATA_NUM_BYTES {64} \
] [get_ips dwc_strm_sink]

create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name dwc_strm_source
set_property -dict [list \
  CONFIG.HAS_TKEEP {1} \
  CONFIG.HAS_TLAST {1} \
  CONFIG.M_TDATA_NUM_BYTES {64} \
  CONFIG.S_TDATA_NUM_BYTES {${OLEN_BYTES}} \
] [get_ips dwc_strm_source]

# AXI regs
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_reg_32
set_property CONFIG.TDATA_NUM_BYTES {4} [get_ips axis_reg_32]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_8
set_property -dict [list CONFIG.TDATA_NUM_BYTES {1} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_8]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_16
set_property -dict [list CONFIG.TDATA_NUM_BYTES {2} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_16]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_32
set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_32]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_64
set_property -dict [list CONFIG.TDATA_NUM_BYTES {8} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_64]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_128
set_property -dict [list CONFIG.TDATA_NUM_BYTES {16} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_128]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_256
set_property -dict [list CONFIG.TDATA_NUM_BYTES {32} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_256]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_512
set_property -dict [list CONFIG.TDATA_NUM_BYTES {64} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_512]


create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_8
set_property -dict [list CONFIG.TDATA_NUM_BYTES {1} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_8]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_16
set_property -dict [list CONFIG.TDATA_NUM_BYTES {2} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_16]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_32
set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_32]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_64
set_property -dict [list CONFIG.TDATA_NUM_BYTES {8} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_64]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_128
set_property -dict [list CONFIG.TDATA_NUM_BYTES {16} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_128]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_256
set_property -dict [list CONFIG.TDATA_NUM_BYTES {32} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_256]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_512
set_property -dict [list CONFIG.TDATA_NUM_BYTES {64} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_512]


create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axi_register_slice_256
set_property -dict [list CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {256} CONFIG.HAS_QOS {0} CONFIG.HAS_REGION {0} CONFIG.REG_AW {1} CONFIG.REG_AR {1} CONFIG.REG_B {1} CONFIG.ID_WIDTH {2} CONFIG.MAX_BURST_LENGTH {14} CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.NUM_WRITE_OUTSTANDING {32}] [get_ips axi_register_slice_256]

create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axi_register_slice_512
set_property -dict [list CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {512} CONFIG.HAS_QOS {0} CONFIG.HAS_REGION {0} CONFIG.REG_AW {1} CONFIG.REG_AR {1} CONFIG.REG_B {1} CONFIG.ID_WIDTH {2} CONFIG.MAX_BURST_LENGTH {14} CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.NUM_WRITE_OUTSTANDING {32}] [get_ips axi_register_slice_512]

create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axil_register_slice_64
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {64} CONFIG.HAS_PROT {0} CONFIG.DATA_WIDTH {64} CONFIG.REG_AW {1} CONFIG.REG_AR {1} CONFIG.REG_W {1} CONFIG.REG_R {1} CONFIG.REG_B {1} ] [get_ips axil_register_slice_64]

# FIFO
create_ip -name axis_data_fifo -vendor xilinx.com -library ip -version 2.0 -module_name axis_data_fifo_slv
set_property -dict [list \
  CONFIG.FIFO_DEPTH {1024} \
  CONFIG.TDATA_NUM_BYTES {8} \
] [get_ips axis_data_fifo_slv]