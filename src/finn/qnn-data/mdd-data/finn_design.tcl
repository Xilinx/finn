# Copyright (c) 2022  Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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
# @brief	Address range defines for FINN IP.
# @author	Thomas B. Preußer <thomas.preusser@amd.com>
##

proc generate {drv_handle} {
	# Bounds of all exposed slave address ranges to xparameters.h
	set file_handle [hsi::utils::open_include_file "xparameters.h"]
	foreach drv [hsi::get_drivers -filter "NAME==[common::get_property NAME $drv_handle]"] {
		generate_memrange_parameters $drv $file_handle
	}
	close $file_handle
}

proc generate_memrange_parameters {drv_handle file_handle} {
	# Collect unique slave interfaces to custom module
	array unset ranges
	foreach mem_range [hsi::get_mem_ranges -of_object [hsi::get_cells -hier [hsi::get_sw_processor]] $drv_handle] {
		set ranges([common::get_property SLAVE_INTERFACE $mem_range]) [list \
			[common::get_property BASE_NAME  $mem_range] \
			[common::get_property BASE_VALUE $mem_range] \
			[common::get_property HIGH_NAME  $mem_range] \
			[common::get_property HIGH_VALUE $mem_range] \
		]
	}

	# Produce defines for the address range bounds
	set prefix "XPAR_[string toupper $drv_handle]"
	foreach {key val} [array get ranges] {
		puts $file_handle "#define [format "%s_%s_%s" $prefix $key [lindex $val 0]] [lindex $val 1]"
		puts $file_handle "#define [format "%s_%s_%s" $prefix $key [lindex $val 2]] [lindex $val 3]"
	}
	puts $file_handle ""
}
