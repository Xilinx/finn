# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "CHECKSUM_COUNT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SIG_APPLICATION" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SIG_CUSTOMER" -parent ${Page_0}
  ipgui::add_param $IPINST -name "VERSION" -parent ${Page_0}


}

proc update_PARAM_VALUE.CHECKSUM_COUNT { PARAM_VALUE.CHECKSUM_COUNT } {
	# Procedure called to update CHECKSUM_COUNT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CHECKSUM_COUNT { PARAM_VALUE.CHECKSUM_COUNT } {
	# Procedure called to validate CHECKSUM_COUNT
	return true
}

proc update_PARAM_VALUE.SIG_APPLICATION { PARAM_VALUE.SIG_APPLICATION } {
	# Procedure called to update SIG_APPLICATION when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SIG_APPLICATION { PARAM_VALUE.SIG_APPLICATION } {
	# Procedure called to validate SIG_APPLICATION
	return true
}

proc update_PARAM_VALUE.SIG_CUSTOMER { PARAM_VALUE.SIG_CUSTOMER } {
	# Procedure called to update SIG_CUSTOMER when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SIG_CUSTOMER { PARAM_VALUE.SIG_CUSTOMER } {
	# Procedure called to validate SIG_CUSTOMER
	return true
}

proc update_PARAM_VALUE.VERSION { PARAM_VALUE.VERSION } {
	# Procedure called to update VERSION when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.VERSION { PARAM_VALUE.VERSION } {
	# Procedure called to validate VERSION
	return true
}


proc update_MODELPARAM_VALUE.SIG_CUSTOMER { MODELPARAM_VALUE.SIG_CUSTOMER PARAM_VALUE.SIG_CUSTOMER } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SIG_CUSTOMER}] ${MODELPARAM_VALUE.SIG_CUSTOMER}
}

proc update_MODELPARAM_VALUE.SIG_APPLICATION { MODELPARAM_VALUE.SIG_APPLICATION PARAM_VALUE.SIG_APPLICATION } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SIG_APPLICATION}] ${MODELPARAM_VALUE.SIG_APPLICATION}
}

proc update_MODELPARAM_VALUE.VERSION { MODELPARAM_VALUE.VERSION PARAM_VALUE.VERSION } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.VERSION}] ${MODELPARAM_VALUE.VERSION}
}

proc update_MODELPARAM_VALUE.CHECKSUM_COUNT { MODELPARAM_VALUE.CHECKSUM_COUNT PARAM_VALUE.CHECKSUM_COUNT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CHECKSUM_COUNT}] ${MODELPARAM_VALUE.CHECKSUM_COUNT}
}

