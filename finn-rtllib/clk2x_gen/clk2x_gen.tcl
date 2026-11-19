# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Block design procedures for the internal clock generator of a stitched IP
# (CreateStitchedIP with clk2x_gen). The generator derives the phase-aligned
# 1x and 2x clocks of the design logic from ap_clk, so that the IP needs no
# ap_clk2x input:
#
#   ap_clk ──► Clocking Wizard (MMCM) ──► clk2x_domain ──► clk, clk2x, rst_n
#   ap_rst_n ─────────────────────────────────┘
#
# clk2x_domain (hdl/clk2x_domain.v) forwards the clocks, holds the design
# reset rst_n until the MMCM has locked, and replaces the MMCM by ideal
# clocks in simulation. Every external AXI interface passes a gate
# (clk2x_gate) that blocks its handshakes while rst_n is asserted.

# Returns the MMCM settings deriving f_mhz (O1) and 2*f_mhz (O2) from an input
# clock of f_mhz on the given part as a dict: M, D, O1, O2, the VCO frequency
# VCO and the input frequency range FIN_MIN..FIN_MAX that keeps the VCO within
# its operating range.
#
# The VCO runs at f_mhz*O1 with O1 even, so that O2 = O1/2 is an integer. O1
# is chosen to put the VCO closest to the geometric center of its range: this
# maximizes the tolerated deviation of the actual from the configured input
# frequency (both ways, as a ratio), e.g. for a PS or platform clock that
# cannot be synthesized exactly. Beyond FIN_MIN..FIN_MAX, the MMCM settings are
# illegal and Vivado reports PDRC-179 on the actual input clock. D only keeps
# the phase detector frequency below its limit and does not affect the VCO.
proc clk2x_mmcm_params {part f_mhz} {
    set arch [get_property ARCHITECTURE [get_parts $part]]
    # VCO range (MHz), maximum M, minimum O1 by MMCM generation (data sheets)
    if {[string match versal* $arch]} {
        # MMCME5: O2 >= 2
        lassign {2160.0 4320.0 432 4} vco_min vco_max m_max o1_min
    } elseif {[string match *uplus* $arch]} {
        # UltraScale+: MMCME4
        lassign {800.0 1600.0 128 2} vco_min vco_max m_max o1_min
    } elseif {[regexp {u$} $arch]} {
        # UltraScale: MMCME3
        lassign {600.0 1440.0 64 2} vco_min vco_max m_max o1_min
    } else {
        # 7-series: MMCME2, maximum VCO frequency by speed grade
        if {![regexp {(\d)} [get_property SPEED [get_parts $part]] -> grade]} {
            set grade 1
        }
        set vco_max [expr {$grade >= 3 ? 1600.0 : ($grade == 2 ? 1440.0 : 1200.0)}]
        lassign [list 600.0 $vco_max 64 2] vco_min vco_max m_max o1_min
    }
    set pfd_max 450.0
    set center [expr {sqrt($vco_min * $vco_max)}]
    set best ""
    for {set o1 $o1_min} {$f_mhz * $o1 <= $vco_max} {incr o1 2} {
        set vco [expr {$f_mhz * $o1}]
        if {$vco < $vco_min} {
            continue
        }
        set dist [expr {abs(log($vco / $center))}]
        if {$best eq "" || $dist < [lindex $best 0]} {
            set best [list $dist $o1]
        }
    }
    if {$best eq ""} {
        error "clk2x_gen: no MMCM setting on $part ($arch) derives $f_mhz and [expr {2 * $f_mhz}]\
            MHz from $f_mhz MHz (VCO $vco_min..$vco_max MHz)"
    }
    set o1 [lindex $best 1]
    set d [expr {int(ceil($f_mhz / $pfd_max))}]
    set m [expr {$o1 * $d}]
    if {$m > $m_max} {
        error "clk2x_gen: the MMCM on $part cannot multiply $f_mhz MHz by $m"
    }
    return [dict create M $m D $d O1 $o1 O2 [expr {$o1 / 2}] VCO [expr {$f_mhz * $o1}] \
        FIN_MIN [expr {$vco_min / $o1}] FIN_MAX [expr {$vco_max / $o1}]]
}

# Creates the ports ap_clk and ap_rst_n, the Clocking Wizard clk2x_gen and
# clk2x_domain (whose source must have been added to the project) for an
# input clock of f_mhz. The design logic connects to clk2x_domain/clk, clk2x
# and rst_n. Writes the MMCM settings as JSON to json_file and the clock
# constraint of the generator to xdc_file, see clk2x_gen_xdc, and adds it to the
# project scoped to the current block design. (The IP packager does not pick it
# up from there.)
proc clk2x_gen_create {f_mhz json_file xdc_file group} {
    set part [get_property PART [current_project]]
    set p [clk2x_mmcm_params $part $f_mhz]
    set f1 [format %.3f $f_mhz]
    set f2 [format %.3f [expr {2 * $f_mhz}]]

    create_bd_port -dir I -type clk -freq_hz [expr {round($f_mhz * 1e6)}] ap_clk
    create_bd_port -dir I -type rst ap_rst_n
    set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports ap_rst_n]

    # Request the frequencies first, then override the wizard's choice of
    # M/D/O by the centered one.
    if {[string match versal* [get_property ARCHITECTURE [get_parts $part]]]} {
        set gen [create_bd_cell -type ip -vlnv [get_ipdefs -all xilinx.com:ip:clk_wizard:*] clk2x_gen]
        set_property -dict [list CONFIG.PRIM_IN_FREQ $f1 CONFIG.PRIM_SOURCE {No_buffer} \
            CONFIG.PRIMITIVE_TYPE {MMCM} \
            CONFIG.CLKOUT_USED {true,true,false,false,false,false,false} \
            CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY "$f1,$f2,100.000,100.000,100.000,100.000,100.000" \
            CONFIG.CLKOUT_REQUESTED_PHASE {0,0,0,0,0,0,0} \
            CONFIG.USE_PHASE_ALIGNMENT {true} CONFIG.USE_LOCKED {true} \
            CONFIG.USE_RESET {true} CONFIG.RESET_TYPE {ACTIVE_LOW}] $gen
        set_property -dict [list CONFIG.OVERRIDE_PRIMITIVE {true} \
            CONFIG.CLKFBOUT_MULT [dict get $p M] CONFIG.DIVCLK_DIVIDE [dict get $p D] \
            CONFIG.CLKOUT1_DIVIDE [dict get $p O1] CONFIG.CLKOUT2_DIVIDE [dict get $p O2]] $gen
    } else {
        set gen [create_bd_cell -type ip -vlnv [get_ipdefs -all xilinx.com:ip:clk_wiz:*] clk2x_gen]
        set_property -dict [list CONFIG.PRIM_IN_FREQ $f1 CONFIG.PRIM_SOURCE {No_buffer} \
            CONFIG.NUM_OUT_CLKS {2} CONFIG.CLKOUT2_USED {true} \
            CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $f1 CONFIG.CLKOUT2_REQUESTED_OUT_FREQ $f2 \
            CONFIG.CLKOUT1_REQUESTED_PHASE {0} CONFIG.CLKOUT2_REQUESTED_PHASE {0} \
            CONFIG.USE_PHASE_ALIGNMENT {true} CONFIG.USE_LOCKED {true} \
            CONFIG.USE_RESET {true} CONFIG.RESET_TYPE {ACTIVE_LOW}] $gen
        set_property -dict [list CONFIG.OVERRIDE_MMCM {true} \
            CONFIG.MMCM_CLKFBOUT_MULT_F [dict get $p M] \
            CONFIG.MMCM_DIVCLK_DIVIDE [dict get $p D] \
            CONFIG.MMCM_CLKOUT0_DIVIDE_F [dict get $p O1] \
            CONFIG.MMCM_CLKOUT1_DIVIDE [dict get $p O2]] $gen
    }
    # The output clocks of clk2x_domain get their FREQ_HZ below: a FREQ_HZ in its
    # source would be read-only, so that the IP packager misses it.
    set_msg_config -id {[IP_Flow 19-4751]} -string {FREQ_HZ bus parameter is missing} \
        -new_severity INFO
    set dom [create_bd_cell -type module -reference clk2x_domain clk2x_domain]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins $gen/clk_in1] [get_bd_pins $dom/ap_clk]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins $dom/ap_rst_n]
    connect_bd_net [get_bd_pins $dom/gen_resetn] [get_bd_pins $gen/resetn]
    connect_bd_net [get_bd_pins $gen/clk_out1] [get_bd_pins $dom/gen_clk]
    connect_bd_net [get_bd_pins $gen/clk_out2] [get_bd_pins $dom/gen_clk2x]
    connect_bd_net [get_bd_pins $gen/locked] [get_bd_pins $dom/gen_locked]
    # clk reproduces ap_clk (same frequency, phase-aligned): one clock domain,
    # so that the external interfaces, which are clocked by clk, can be
    # associated with ap_clk
    set_property -dict [list CONFIG.FREQ_HZ [expr {round($f_mhz * 1e6)}] \
        CONFIG.CLK_DOMAIN [get_property CONFIG.CLK_DOMAIN [get_bd_ports ap_clk]]] \
        [get_bd_pins $dom/clk]
    set_property CONFIG.FREQ_HZ [expr {round(2 * $f_mhz * 1e6)}] [get_bd_pins $dom/clk2x]

    set fp [open $json_file w]
    puts $fp "\{\"f_mhz\": $f_mhz, \"part\": \"$part\""
    dict for {k v} $p {
        puts $fp ", \"$k\": $v"
    }
    puts $fp "\}"
    close $fp
    clk2x_gen_xdc $xdc_file $group
    add_files -fileset constrs_1 -norecurse $xdc_file
    set_property SCOPED_TO_REF [get_property NAME [current_bd_design]] [get_files $xdc_file]
    puts "INFO: \[FINN\] clk2x_gen: MMCM M=[dict get $p M] D=[dict get $p D]\
        O1=[dict get $p O1] O2=[dict get $p O2], VCO [dict get $p VCO] MHz, ap_clk\
        $f_mhz MHz (tolerated [format %.2f [dict get $p FIN_MIN]]..[format %.2f\
        [dict get $p FIN_MAX]] MHz)"
}

# Writes the clock constraint of the generator to xdc_file: its 1x and 2x clock
# nets form one CLOCK_DELAY_GROUP, so that the router balances their clock
# trees. Otherwise, their skew easily takes a large part of the 2x period from
# the synchronous crossings between both domains. The constraint is meant to be
# scoped to the stitched block design (SCOPED_TO_REF) or to the packaged IP. The
# group name should be unique per generator: a group balances the clocks of one
# clock source.
proc clk2x_gen_xdc {xdc_file group} {
    set fp [open $xdc_file w]
    puts $fp "# FINN clk2x_gen: balance the clock trees of the 1x and 2x clocks."
    puts $fp "set_property CLOCK_DELAY_GROUP $group \[get_nets -of_objects\
        \[get_pins -hierarchical -filter {NAME =~ *clk2x_gen/clk_out*}\]\]"
    close $fp
}

# Makes the interface pin intf_pin of a block design cell external as port_name
# through a gate that blocks its handshakes while the design reset
# clk2x_domain/rst_n is asserted: every VALID and READY signal is ANDed with
# rst_n, in both directions. The gate module is generated into src_dir from
# the signals of intf_pin, so that the external interface has exactly the
# signals of the cell's. Returns the gate cell.
#
# Without the gate, the system could exchange data with the design between
# the release of ap_rst_n and that of rst_n (MMCM lock): a stream slave in
# reset may signal TREADY, and the state of the design logic while its clock
# is stopped (MMCM in reset) is not reset at all. The gate is combinational,
# so that it closes immediately when rst_n is asserted.
#
# For an AXI-Lite slave, a blocked transfer stalls the master (no READY) until
# the design leaves reset, rather than failing with an error response. This
# preserves the transfer and is AXI compliant. The stall is bounded by the
# lock time of the MMCM after ap_rst_n is released.
#
# A gate ends the address path of the block design: an AXI-MM master's
# external segment is <gate>/outer/SEG_<port_name>_Reg.
proc clk2x_gate {intf_pin port_name src_dir} {
    set vlnv [string map {_rtl: :} [get_property VLNV $intf_pin]]
    set mode [string tolower [get_property MODE $intf_pin]]
    set inner_mode [expr {$mode eq "master" ? "slave" : "master"}]
    set module "clk2x_gate_$port_name"
    set ports [list]
    set assigns [list]
    set logicals [list]
    foreach pin [get_bd_pins -of_objects $intf_pin] {
        # the logical name is the suffix of the physical one: m_axi_gmem_AWADDR, in0_V_TDATA
        set logical [string toupper [lindex [split [get_property NAME $pin] _] end]]
        if {[lsearch -exact $logicals $logical] >= 0} {
            error "clk2x_gate: ambiguous signal $logical of $intf_pin"
        }
        lappend logicals $logical
        set lc [string tolower $logical]
        set left [get_property LEFT $pin]
        set range [expr {$left eq "" ? "" : "\[$left:[get_property RIGHT $pin]\]  "}]
        if {[get_property DIR $pin] eq "O"} {
            lassign {input output inner outer} inner_dir outer_dir src dst
        } else {
            lassign {output input outer inner} inner_dir outer_dir src dst
        }
        set inner_attr "(* X_INTERFACE_INFO = \"$vlnv inner $logical\" *)"
        set outer_attr "(* X_INTERFACE_INFO = \"$vlnv outer $logical\" *)"
        if {[llength $logicals] == 1} {
            set inner_attr "(* X_INTERFACE_MODE = \"$inner_mode\" *) $inner_attr"
            set outer_attr "(* X_INTERFACE_MODE = \"$mode\" *) $outer_attr"
        }
        lappend ports "\t$inner_attr\n\t$inner_dir\t${range}inner_$lc"
        lappend ports "\t$outer_attr\n\t$outer_dir\t${range}outer_$lc"
        if {[regexp {(VALID|READY)$} $logical]} {
            lappend assigns "\tassign\t${dst}_$lc = ${src}_$lc && rst_n;"
        } else {
            lappend assigns "\tassign\t${dst}_$lc = ${src}_$lc;"
        }
    }

    set src "$src_dir/$module.v"
    set fp [open $src w]
    puts $fp "// Generated by clk2x_gate (finn-rtllib/clk2x_gen/clk2x_gen.tcl) for $intf_pin:"
    puts $fp "// blocks the VALID/READY handshakes while rst_n is asserted."
    puts $fp "module $module ("
    puts $fp "\t(* X_INTERFACE_INFO = \"xilinx.com:signal:clock:1.0 clk CLK\" *)"
    puts $fp "\t(* X_INTERFACE_PARAMETER = \"ASSOCIATED_BUSIF inner:outer, ASSOCIATED_RESET rst_n\" *)"
    puts $fp "\tinput\tclk,"
    puts $fp "\t(* X_INTERFACE_INFO = \"xilinx.com:signal:reset:1.0 rst_n RST\" *)"
    puts $fp "\t(* X_INTERFACE_PARAMETER = \"POLARITY ACTIVE_LOW\" *)"
    puts $fp "\tinput\trst_n,"
    puts $fp [join $ports ",\n"]
    puts $fp ");"
    puts $fp [join $assigns "\n"]
    puts $fp "endmodule"
    close $fp
    add_files -norecurse $src

    set gate [create_bd_cell -type module -reference $module $module]
    connect_bd_net [get_bd_pins clk2x_domain/clk] [get_bd_pins $gate/clk]
    connect_bd_net [get_bd_pins clk2x_domain/rst_n] [get_bd_pins $gate/rst_n]
    connect_bd_intf_net $intf_pin [get_bd_intf_pins $gate/inner]
    make_bd_intf_pins_external [get_bd_intf_pins $gate/outer]
    set_property NAME $port_name \
        [get_bd_intf_ports -of_objects [get_bd_intf_nets -of_objects [get_bd_intf_pins $gate/outer]]]
    return $gate
}
