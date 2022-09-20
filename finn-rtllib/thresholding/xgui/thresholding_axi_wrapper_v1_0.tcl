# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  set C [ipgui::add_param $IPINST -name "C" -parent ${Page_0}]
  set_property tooltip {Channel Count} ${C}
  set C_BITS [ipgui::add_param $IPINST -name "C_BITS" -parent ${Page_0}]
  set_property tooltip {Must be clog2(C)} ${C_BITS}
  set M [ipgui::add_param $IPINST -name "M" -parent ${Page_0}]
  set_property tooltip {Input Precision} ${M}
  set N [ipgui::add_param $IPINST -name "N" -parent ${Page_0}]
  set_property tooltip {Output Precision} ${N}


}

proc update_PARAM_VALUE.C { PARAM_VALUE.C } {
	# Procedure called to update C when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C { PARAM_VALUE.C } {
	# Procedure called to validate C
	return true
}

proc update_PARAM_VALUE.C_BITS { PARAM_VALUE.C_BITS } {
	# Procedure called to update C_BITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_BITS { PARAM_VALUE.C_BITS } {
	# Procedure called to validate C_BITS
	return true
}

proc update_PARAM_VALUE.M { PARAM_VALUE.M } {
	# Procedure called to update M when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.M { PARAM_VALUE.M } {
	# Procedure called to validate M
	return true
}

proc update_PARAM_VALUE.N { PARAM_VALUE.N } {
	# Procedure called to update N when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.N { PARAM_VALUE.N } {
	# Procedure called to validate N
	return true
}


proc update_MODELPARAM_VALUE.N { MODELPARAM_VALUE.N PARAM_VALUE.N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.N}] ${MODELPARAM_VALUE.N}
}

proc update_MODELPARAM_VALUE.M { MODELPARAM_VALUE.M PARAM_VALUE.M } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.M}] ${MODELPARAM_VALUE.M}
}

proc update_MODELPARAM_VALUE.C { MODELPARAM_VALUE.C PARAM_VALUE.C } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C}] ${MODELPARAM_VALUE.C}
}

proc update_MODELPARAM_VALUE.C_BITS { MODELPARAM_VALUE.C_BITS PARAM_VALUE.C_BITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_BITS}] ${MODELPARAM_VALUE.C_BITS}
}

