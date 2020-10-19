
# Loading additional proc with user specified bodies to compute parameter values.
source [file join [file dirname [file dirname [info script]]] gui/memstream_v1_0.gtcl]

# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "AXILITE_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CONFIG_EN" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MEM_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MEM_INIT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MEM_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "NSTREAMS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "RAM_STYLE" -parent ${Page_0} -widget comboBox
  ipgui::add_param $IPINST -name "STRM0_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM0_OFFSET" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM0_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM1_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM1_OFFSET" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM1_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM2_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM2_OFFSET" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM2_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM3_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM3_OFFSET" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM3_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM4_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM4_OFFSET" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM4_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM5_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM5_OFFSET" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRM5_WIDTH" -parent ${Page_0}


}

proc update_PARAM_VALUE.AXILITE_ADDR_WIDTH { PARAM_VALUE.AXILITE_ADDR_WIDTH PARAM_VALUE.MEM_DEPTH PARAM_VALUE.MEM_WIDTH } {
	# Procedure called to update AXILITE_ADDR_WIDTH when any of the dependent parameters in the arguments change
	
	set AXILITE_ADDR_WIDTH ${PARAM_VALUE.AXILITE_ADDR_WIDTH}
	set MEM_DEPTH ${PARAM_VALUE.MEM_DEPTH}
	set MEM_WIDTH ${PARAM_VALUE.MEM_WIDTH}
	set values(MEM_DEPTH) [get_property value $MEM_DEPTH]
	set values(MEM_WIDTH) [get_property value $MEM_WIDTH]
	set_property value [gen_USERPARAMETER_AXILITE_ADDR_WIDTH_VALUE $values(MEM_DEPTH) $values(MEM_WIDTH)] $AXILITE_ADDR_WIDTH
}

proc validate_PARAM_VALUE.AXILITE_ADDR_WIDTH { PARAM_VALUE.AXILITE_ADDR_WIDTH } {
	# Procedure called to validate AXILITE_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.CONFIG_EN { PARAM_VALUE.CONFIG_EN } {
	# Procedure called to update CONFIG_EN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONFIG_EN { PARAM_VALUE.CONFIG_EN } {
	# Procedure called to validate CONFIG_EN
	return true
}

proc update_PARAM_VALUE.MEM_DEPTH { PARAM_VALUE.MEM_DEPTH } {
	# Procedure called to update MEM_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MEM_DEPTH { PARAM_VALUE.MEM_DEPTH } {
	# Procedure called to validate MEM_DEPTH
	return true
}

proc update_PARAM_VALUE.MEM_INIT { PARAM_VALUE.MEM_INIT } {
	# Procedure called to update MEM_INIT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MEM_INIT { PARAM_VALUE.MEM_INIT } {
	# Procedure called to validate MEM_INIT
	return true
}

proc update_PARAM_VALUE.MEM_WIDTH { PARAM_VALUE.MEM_WIDTH } {
	# Procedure called to update MEM_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MEM_WIDTH { PARAM_VALUE.MEM_WIDTH } {
	# Procedure called to validate MEM_WIDTH
	return true
}

proc update_PARAM_VALUE.NSTREAMS { PARAM_VALUE.NSTREAMS } {
	# Procedure called to update NSTREAMS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NSTREAMS { PARAM_VALUE.NSTREAMS } {
	# Procedure called to validate NSTREAMS
	return true
}

proc update_PARAM_VALUE.RAM_STYLE { PARAM_VALUE.RAM_STYLE } {
	# Procedure called to update RAM_STYLE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.RAM_STYLE { PARAM_VALUE.RAM_STYLE } {
	# Procedure called to validate RAM_STYLE
	return true
}

proc update_PARAM_VALUE.STRM0_DEPTH { PARAM_VALUE.STRM0_DEPTH } {
	# Procedure called to update STRM0_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM0_DEPTH { PARAM_VALUE.STRM0_DEPTH } {
	# Procedure called to validate STRM0_DEPTH
	return true
}

proc update_PARAM_VALUE.STRM0_OFFSET { PARAM_VALUE.STRM0_OFFSET } {
	# Procedure called to update STRM0_OFFSET when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM0_OFFSET { PARAM_VALUE.STRM0_OFFSET } {
	# Procedure called to validate STRM0_OFFSET
	return true
}

proc update_PARAM_VALUE.STRM0_WIDTH { PARAM_VALUE.STRM0_WIDTH } {
	# Procedure called to update STRM0_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM0_WIDTH { PARAM_VALUE.STRM0_WIDTH } {
	# Procedure called to validate STRM0_WIDTH
	return true
}

proc update_PARAM_VALUE.STRM1_DEPTH { PARAM_VALUE.STRM1_DEPTH } {
	# Procedure called to update STRM1_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM1_DEPTH { PARAM_VALUE.STRM1_DEPTH } {
	# Procedure called to validate STRM1_DEPTH
	return true
}

proc update_PARAM_VALUE.STRM1_OFFSET { PARAM_VALUE.STRM1_OFFSET } {
	# Procedure called to update STRM1_OFFSET when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM1_OFFSET { PARAM_VALUE.STRM1_OFFSET } {
	# Procedure called to validate STRM1_OFFSET
	return true
}

proc update_PARAM_VALUE.STRM1_WIDTH { PARAM_VALUE.STRM1_WIDTH } {
	# Procedure called to update STRM1_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM1_WIDTH { PARAM_VALUE.STRM1_WIDTH } {
	# Procedure called to validate STRM1_WIDTH
	return true
}

proc update_PARAM_VALUE.STRM2_DEPTH { PARAM_VALUE.STRM2_DEPTH } {
	# Procedure called to update STRM2_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM2_DEPTH { PARAM_VALUE.STRM2_DEPTH } {
	# Procedure called to validate STRM2_DEPTH
	return true
}

proc update_PARAM_VALUE.STRM2_OFFSET { PARAM_VALUE.STRM2_OFFSET } {
	# Procedure called to update STRM2_OFFSET when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM2_OFFSET { PARAM_VALUE.STRM2_OFFSET } {
	# Procedure called to validate STRM2_OFFSET
	return true
}

proc update_PARAM_VALUE.STRM2_WIDTH { PARAM_VALUE.STRM2_WIDTH } {
	# Procedure called to update STRM2_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM2_WIDTH { PARAM_VALUE.STRM2_WIDTH } {
	# Procedure called to validate STRM2_WIDTH
	return true
}

proc update_PARAM_VALUE.STRM3_DEPTH { PARAM_VALUE.STRM3_DEPTH } {
	# Procedure called to update STRM3_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM3_DEPTH { PARAM_VALUE.STRM3_DEPTH } {
	# Procedure called to validate STRM3_DEPTH
	return true
}

proc update_PARAM_VALUE.STRM3_OFFSET { PARAM_VALUE.STRM3_OFFSET } {
	# Procedure called to update STRM3_OFFSET when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM3_OFFSET { PARAM_VALUE.STRM3_OFFSET } {
	# Procedure called to validate STRM3_OFFSET
	return true
}

proc update_PARAM_VALUE.STRM3_WIDTH { PARAM_VALUE.STRM3_WIDTH } {
	# Procedure called to update STRM3_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM3_WIDTH { PARAM_VALUE.STRM3_WIDTH } {
	# Procedure called to validate STRM3_WIDTH
	return true
}

proc update_PARAM_VALUE.STRM4_DEPTH { PARAM_VALUE.STRM4_DEPTH } {
	# Procedure called to update STRM4_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM4_DEPTH { PARAM_VALUE.STRM4_DEPTH } {
	# Procedure called to validate STRM4_DEPTH
	return true
}

proc update_PARAM_VALUE.STRM4_OFFSET { PARAM_VALUE.STRM4_OFFSET } {
	# Procedure called to update STRM4_OFFSET when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM4_OFFSET { PARAM_VALUE.STRM4_OFFSET } {
	# Procedure called to validate STRM4_OFFSET
	return true
}

proc update_PARAM_VALUE.STRM4_WIDTH { PARAM_VALUE.STRM4_WIDTH } {
	# Procedure called to update STRM4_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM4_WIDTH { PARAM_VALUE.STRM4_WIDTH } {
	# Procedure called to validate STRM4_WIDTH
	return true
}

proc update_PARAM_VALUE.STRM5_DEPTH { PARAM_VALUE.STRM5_DEPTH } {
	# Procedure called to update STRM5_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM5_DEPTH { PARAM_VALUE.STRM5_DEPTH } {
	# Procedure called to validate STRM5_DEPTH
	return true
}

proc update_PARAM_VALUE.STRM5_OFFSET { PARAM_VALUE.STRM5_OFFSET } {
	# Procedure called to update STRM5_OFFSET when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM5_OFFSET { PARAM_VALUE.STRM5_OFFSET } {
	# Procedure called to validate STRM5_OFFSET
	return true
}

proc update_PARAM_VALUE.STRM5_WIDTH { PARAM_VALUE.STRM5_WIDTH } {
	# Procedure called to update STRM5_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRM5_WIDTH { PARAM_VALUE.STRM5_WIDTH } {
	# Procedure called to validate STRM5_WIDTH
	return true
}


proc update_MODELPARAM_VALUE.CONFIG_EN { MODELPARAM_VALUE.CONFIG_EN PARAM_VALUE.CONFIG_EN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONFIG_EN}] ${MODELPARAM_VALUE.CONFIG_EN}
}

proc update_MODELPARAM_VALUE.NSTREAMS { MODELPARAM_VALUE.NSTREAMS PARAM_VALUE.NSTREAMS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NSTREAMS}] ${MODELPARAM_VALUE.NSTREAMS}
}

proc update_MODELPARAM_VALUE.MEM_DEPTH { MODELPARAM_VALUE.MEM_DEPTH PARAM_VALUE.MEM_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MEM_DEPTH}] ${MODELPARAM_VALUE.MEM_DEPTH}
}

proc update_MODELPARAM_VALUE.MEM_WIDTH { MODELPARAM_VALUE.MEM_WIDTH PARAM_VALUE.MEM_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MEM_WIDTH}] ${MODELPARAM_VALUE.MEM_WIDTH}
}

proc update_MODELPARAM_VALUE.MEM_INIT { MODELPARAM_VALUE.MEM_INIT PARAM_VALUE.MEM_INIT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MEM_INIT}] ${MODELPARAM_VALUE.MEM_INIT}
}

proc update_MODELPARAM_VALUE.RAM_STYLE { MODELPARAM_VALUE.RAM_STYLE PARAM_VALUE.RAM_STYLE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.RAM_STYLE}] ${MODELPARAM_VALUE.RAM_STYLE}
}

proc update_MODELPARAM_VALUE.STRM0_WIDTH { MODELPARAM_VALUE.STRM0_WIDTH PARAM_VALUE.STRM0_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM0_WIDTH}] ${MODELPARAM_VALUE.STRM0_WIDTH}
}

proc update_MODELPARAM_VALUE.STRM1_WIDTH { MODELPARAM_VALUE.STRM1_WIDTH PARAM_VALUE.STRM1_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM1_WIDTH}] ${MODELPARAM_VALUE.STRM1_WIDTH}
}

proc update_MODELPARAM_VALUE.STRM2_WIDTH { MODELPARAM_VALUE.STRM2_WIDTH PARAM_VALUE.STRM2_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM2_WIDTH}] ${MODELPARAM_VALUE.STRM2_WIDTH}
}

proc update_MODELPARAM_VALUE.STRM3_WIDTH { MODELPARAM_VALUE.STRM3_WIDTH PARAM_VALUE.STRM3_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM3_WIDTH}] ${MODELPARAM_VALUE.STRM3_WIDTH}
}

proc update_MODELPARAM_VALUE.STRM4_WIDTH { MODELPARAM_VALUE.STRM4_WIDTH PARAM_VALUE.STRM4_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM4_WIDTH}] ${MODELPARAM_VALUE.STRM4_WIDTH}
}

proc update_MODELPARAM_VALUE.STRM5_WIDTH { MODELPARAM_VALUE.STRM5_WIDTH PARAM_VALUE.STRM5_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM5_WIDTH}] ${MODELPARAM_VALUE.STRM5_WIDTH}
}

proc update_MODELPARAM_VALUE.STRM0_DEPTH { MODELPARAM_VALUE.STRM0_DEPTH PARAM_VALUE.STRM0_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM0_DEPTH}] ${MODELPARAM_VALUE.STRM0_DEPTH}
}

proc update_MODELPARAM_VALUE.STRM1_DEPTH { MODELPARAM_VALUE.STRM1_DEPTH PARAM_VALUE.STRM1_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM1_DEPTH}] ${MODELPARAM_VALUE.STRM1_DEPTH}
}

proc update_MODELPARAM_VALUE.STRM2_DEPTH { MODELPARAM_VALUE.STRM2_DEPTH PARAM_VALUE.STRM2_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM2_DEPTH}] ${MODELPARAM_VALUE.STRM2_DEPTH}
}

proc update_MODELPARAM_VALUE.STRM3_DEPTH { MODELPARAM_VALUE.STRM3_DEPTH PARAM_VALUE.STRM3_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM3_DEPTH}] ${MODELPARAM_VALUE.STRM3_DEPTH}
}

proc update_MODELPARAM_VALUE.STRM4_DEPTH { MODELPARAM_VALUE.STRM4_DEPTH PARAM_VALUE.STRM4_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM4_DEPTH}] ${MODELPARAM_VALUE.STRM4_DEPTH}
}

proc update_MODELPARAM_VALUE.STRM5_DEPTH { MODELPARAM_VALUE.STRM5_DEPTH PARAM_VALUE.STRM5_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM5_DEPTH}] ${MODELPARAM_VALUE.STRM5_DEPTH}
}

proc update_MODELPARAM_VALUE.STRM0_OFFSET { MODELPARAM_VALUE.STRM0_OFFSET PARAM_VALUE.STRM0_OFFSET } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM0_OFFSET}] ${MODELPARAM_VALUE.STRM0_OFFSET}
}

proc update_MODELPARAM_VALUE.STRM1_OFFSET { MODELPARAM_VALUE.STRM1_OFFSET PARAM_VALUE.STRM1_OFFSET } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM1_OFFSET}] ${MODELPARAM_VALUE.STRM1_OFFSET}
}

proc update_MODELPARAM_VALUE.STRM2_OFFSET { MODELPARAM_VALUE.STRM2_OFFSET PARAM_VALUE.STRM2_OFFSET } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM2_OFFSET}] ${MODELPARAM_VALUE.STRM2_OFFSET}
}

proc update_MODELPARAM_VALUE.STRM3_OFFSET { MODELPARAM_VALUE.STRM3_OFFSET PARAM_VALUE.STRM3_OFFSET } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM3_OFFSET}] ${MODELPARAM_VALUE.STRM3_OFFSET}
}

proc update_MODELPARAM_VALUE.STRM4_OFFSET { MODELPARAM_VALUE.STRM4_OFFSET PARAM_VALUE.STRM4_OFFSET } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM4_OFFSET}] ${MODELPARAM_VALUE.STRM4_OFFSET}
}

proc update_MODELPARAM_VALUE.STRM5_OFFSET { MODELPARAM_VALUE.STRM5_OFFSET PARAM_VALUE.STRM5_OFFSET } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRM5_OFFSET}] ${MODELPARAM_VALUE.STRM5_OFFSET}
}

proc update_MODELPARAM_VALUE.AXILITE_ADDR_WIDTH { MODELPARAM_VALUE.AXILITE_ADDR_WIDTH PARAM_VALUE.AXILITE_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.AXILITE_ADDR_WIDTH}] ${MODELPARAM_VALUE.AXILITE_ADDR_WIDTH}
}

