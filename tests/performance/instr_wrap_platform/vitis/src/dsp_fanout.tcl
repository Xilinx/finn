foreach inst [get_cells -hierarchical -filter {ORIG_REF_NAME == "mvu_vvu_8sx9_dsp58"}] {
	current_instance $inst
	set nets [get_nets -of_objects [get_pins -of_objects [get_cells "genDSPPE[*].genDSPChain[*].genDSP.DSP58_inst"] -filter {BUS_NAME == A || BUS_NAME == B}] -filter  {TYPE == SIGNAL}]
	set_property MAX_FANOUT_MODE CLOCK_REGION $nets
	current_instance -quiet
}
set_property CLOCK_DELAY_GROUP SYNC_CLOCKS [get_nets -of_objects [get_pins -of_objects [get_cells -hierarchical -filter {PRIMITIVE_TYPE == "CLOCK.BUFFER.MBUFGCE"}] -filter {DIRECTION == "out"}]]
