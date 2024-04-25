set PLATFORM_NAME [lindex $argv 0]
set PLATFORM_XSA ${PLATFORM_NAME}_wrapper.xsa
set PLATFORM_PDI ${PLATFORM_NAME}_wrapper.pdi

set HW_SERVER_HOST [lindex $argv 1]
set HW_SERVER_PORT [lindex $argv 2]

# Export the .xsa hardware platform
write_hw_platform -fixed -force $PLATFORM_XSA

# Open the remote device
open_hw_manager
connect_hw_server -url ${HW_SERVER_HOST}:${HW_SERVER_PORT} -allow_non_jtag
current_hw_target [lindex [get_hw_targets] 0]
set_property PARAM.FREQUENCY 15000000 [current_hw_target]
open_hw_target
current_hw_device [lindex [get_hw_devices] 1]
set_property PROGRAM.FILE $PLATFORM_PDI [current_hw_device]
refresh_hw_device -update_hw_probes false [lindex [current_hw_device] 0]

# Program the device
set_property PROBES.FILE {} [current_hw_device]
set_property FULL_PROBES.FILE {} [current_hw_device]
program_hw_devices [current_hw_device]
refresh_hw_device [lindex [current_hw_device] 0]


