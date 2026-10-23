# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# The Vivado side of `finn-platform extract` (finn/platform/extract.py), read-only on
# Vivado's installed part database. Every line it answers starts with CATALOG, so the
# reader ignores Vivado's own; fields are separated by "|".
#
#   vivado -mode batch -source extract.tcl -tclargs parts PATTERN INDEX COUNT
#     the INDEXth of COUNT contiguous slices of the sorted installed parts matching
#     PATTERN (a device's parts are adjacent, and Vivado loads a device once for them):
#     CATALOG PART name|device|package|speed|temperature|architecture|family|
#                  LUT_ELEMENTS|FLIPFLOPS|BLOCK_RAMS|ULTRA_RAMS|DSP|SLRS
#     (an absent property is empty)
#
#   vivado -mode batch -source extract.tcl -tclargs devices PART...
#     for each part, an empty design linked on it (no synthesis), then
#     CATALOG DEVICE part|device
#     CATALOG SLR part|slr|IS_FABRIC|site_type=count,...     (every site of the SLR)
#     CATALOG SLICEM part|bel,...                          (one SLICEM's BELs)
#     CATALOG LINKED part|milliseconds
#     or CATALOG LINK_FAIL part|message

puts "CATALOG TOOL [version -short]|[lindex [split [version] \n] 1]"

proc catalog_parts {pattern index count} {
  set props {NAME DEVICE PACKAGE SPEED TEMPERATURE_GRADE_LETTER ARCHITECTURE FAMILY \
             LUT_ELEMENTS FLIPFLOPS BLOCK_RAMS ULTRA_RAMS DSP SLRS}
  set all [lsort [get_parts -quiet $pattern]]
  puts "CATALOG COUNT [llength $all]"
  set size [expr {([llength $all] + $count - 1) / $count}]
  foreach p [lrange $all [expr {$index * $size}] [expr {($index + 1) * $size - 1}]] {
    puts "CATALOG PART [join [lmap k $props {get_property -quiet $k $p}] |]"
  }
}

proc catalog_device {name} {
  set t0 [clock milliseconds]
  if {[catch {link_design -part $name -name catalog} err]} {
    puts "CATALOG LINK_FAIL $name|[string map {\n { }} $err]"
    return
  }
  puts "CATALOG DEVICE $name|[get_property DEVICE [get_parts $name]]"
  foreach slr [get_slrs -quiet] {
    array unset types
    foreach site [get_sites -quiet -of_objects $slr] {
      set type [get_property SITE_TYPE $site]
      if {[info exists types($type)]} { incr types($type) } else { set types($type) 1 }
    }
    set counts [lmap type [lsort [array names types]] {set _ "$type=$types($type)"}]
    puts "CATALOG SLR $name|$slr|[get_property IS_FABRIC $slr]|[join $counts ,]"
  }
  set slicem [lindex [get_sites -quiet -filter {SITE_TYPE == SLICEM}] 0]
  set bels {}
  if {$slicem ne ""} {
    set bels [lsort [lmap bel [get_bels -quiet -of_objects $slicem] {lindex [split $bel /] end}]]
  }
  puts "CATALOG SLICEM $name|[join $bels ,]"
  close_design
  puts "CATALOG LINKED $name|[expr {[clock milliseconds] - $t0}]"
}

switch -- [lindex $argv 0] {
  parts { catalog_parts [lindex $argv 1] [lindex $argv 2] [lindex $argv 3] }
  devices { foreach name [lrange $argv 1 end] { catalog_device $name } }
  default { puts "CATALOG USAGE parts PATTERN INDEX COUNT | devices PART..."; exit 2 }
}
puts "CATALOG COMPLETE"
exit
