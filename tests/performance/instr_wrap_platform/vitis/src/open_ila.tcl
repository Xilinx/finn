set PLATFORM_NAME [lindex $argv 0]
set PLATFORM_PDI ${PLATFORM_NAME}_wrapper.pdi
set PLATFORM_LTX ${PLATFORM_NAME}_wrapper.ltx

set HW_SERVER_HOST [lindex $argv 1]
set HW_SERVER_PORT [lindex $argv 2]

start_gui

open_hw_manager
connect_hw_server -url ${HW_SERVER_HOST}:${HW_SERVER_PORT} -allow_non_jtag
current_hw_target [lindex [get_hw_targets] 0]
set_property PARAM.FREQUENCY 15000000 [current_hw_target]
open_hw_target
#set_property PROGRAM.FILE $PLATFORM_PDI [current_hw_device]
set_property PROBES.FILE $PLATFORM_LTX [current_hw_device]
set_property FULL_PROBES.FILE $PLATFORM_LTX [current_hw_device]
current_hw_device [lindex [get_hw_devices] 1]
refresh_hw_device [lindex [current_hw_device] 0]

display_hw_ila_data [ get_hw_ila_data hw_ila_data_1 -of_objects [get_hw_ilas -of_objects [current_hw_device] -filter {CELL_NAME=~"*/axis_ila_0"}]]
run_hw_ila [get_hw_ilas -of_objects [current_hw_device] -filter {CELL_NAME=~"*/axis_ila_0"}]
wait_on_hw_ila [get_hw_ilas -of_objects [current_hw_device] -filter {CELL_NAME=~"*/axis_ila_0"}]
display_hw_ila_data [upload_hw_ila_data [get_hw_ilas -of_objects [current_hw_device] -filter {CELL_NAME=~"*/axis_ila_0"}]]
