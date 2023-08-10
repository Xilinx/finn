
# Loading additional proc with user specified bodies to compute parameter values.
source [file join [file dirname [file dirname [info script]]] gui/memstream_v1_0.gtcl]

# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "AXILITE_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "INIT_FILE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "RAM_STYLE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WIDTH" -parent ${Page_0}
}

proc update_PARAM_VALUE.AXILITE_ADDR_WIDTH { PARAM_VALUE.AXILITE_ADDR_WIDTH PARAM_VALUE.DEPTH PARAM_VALUE.WIDTH } {
	# Procedure called to update AXILITE_ADDR_WIDTH when any of the dependent parameters in the arguments change

	set AXILITE_ADDR_WIDTH ${PARAM_VALUE.AXILITE_ADDR_WIDTH}
	set DEPTH ${PARAM_VALUE.DEPTH}
	set WIDTH ${PARAM_VALUE.WIDTH}
	set values(DEPTH) [get_property value $DEPTH]
	set values(WIDTH) [get_property value $WIDTH]
	set_property value [gen_USERPARAMETER_AXILITE_ADDR_WIDTH_VALUE $values(DEPTH) $values(WIDTH)] $AXILITE_ADDR_WIDTH
}

proc validate_PARAM_VALUE.AXILITE_ADDR_WIDTH { PARAM_VALUE.AXILITE_ADDR_WIDTH } {
	# Procedure called to validate AXILITE_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.DEPTH { PARAM_VALUE.DEPTH } {
	# Procedure called to update DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DEPTH { PARAM_VALUE.DEPTH } {
	# Procedure called to validate DEPTH
	return true
}

proc update_PARAM_VALUE.INIT_FILE { PARAM_VALUE.INIT_FILE } {
	# Procedure called to update INIT_FILE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.INIT_FILE { PARAM_VALUE.INIT_FILE } {
	# Procedure called to validate INIT_FILE
	return true
}

proc update_PARAM_VALUE.RAM_STYLE { PARAM_VALUE.RAM_STYLE } {
	# Procedure called to update RAM_STYLE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.RAM_STYLE { PARAM_VALUE.RAM_STYLE } {
	# Procedure called to validate RAM_STYLE
	return true
}

proc update_PARAM_VALUE.WIDTH { PARAM_VALUE.WIDTH } {
	# Procedure called to update WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WIDTH { PARAM_VALUE.WIDTH } {
	# Procedure called to validate WIDTH
	return true
}


proc update_MODELPARAM_VALUE.DEPTH { MODELPARAM_VALUE.DEPTH PARAM_VALUE.DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DEPTH}] ${MODELPARAM_VALUE.DEPTH}
}

proc update_MODELPARAM_VALUE.WIDTH { MODELPARAM_VALUE.WIDTH PARAM_VALUE.WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WIDTH}] ${MODELPARAM_VALUE.WIDTH}
}

proc update_MODELPARAM_VALUE.INIT_FILE { MODELPARAM_VALUE.INIT_FILE PARAM_VALUE.INIT_FILE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.INIT_FILE}] ${MODELPARAM_VALUE.INIT_FILE}
}

proc update_MODELPARAM_VALUE.RAM_STYLE { MODELPARAM_VALUE.RAM_STYLE PARAM_VALUE.RAM_STYLE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.RAM_STYLE}] ${MODELPARAM_VALUE.RAM_STYLE}
}

proc update_MODELPARAM_VALUE.AXILITE_ADDR_WIDTH { MODELPARAM_VALUE.AXILITE_ADDR_WIDTH PARAM_VALUE.AXILITE_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.AXILITE_ADDR_WIDTH}] ${MODELPARAM_VALUE.AXILITE_ADDR_WIDTH}
}
