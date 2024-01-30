
# Loading additional proc with user specified bodies to compute parameter values.
source [file join [file dirname [file dirname [info script]]] gui/thresholding_axi_v1_0.gtcl]

# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "ADDR_BITS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BIAS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CF" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FPARG" -parent ${Page_0}
  ipgui::add_param $IPINST -name "K" -parent ${Page_0}
  ipgui::add_param $IPINST -name "N" -parent ${Page_0}
  ipgui::add_param $IPINST -name "O_BITS" -parent ${Page_0}
  set PE [ipgui::add_param $IPINST -name "PE" -parent ${Page_0}]
  set_property tooltip {PE Count} ${PE}
  ipgui::add_param $IPINST -name "SIGNED" -parent ${Page_0}


}

proc update_PARAM_VALUE.ADDR_BITS { PARAM_VALUE.ADDR_BITS PARAM_VALUE.C PARAM_VALUE.PE PARAM_VALUE.N } {
	# Procedure called to update ADDR_BITS when any of the dependent parameters in the arguments change

	set ADDR_BITS ${PARAM_VALUE.ADDR_BITS}
	set C ${PARAM_VALUE.C}
	set PE ${PARAM_VALUE.PE}
	set N ${PARAM_VALUE.N}
	set values(C) [get_property value $C]
	set values(PE) [get_property value $PE]
	set values(N) [get_property value $N]
	set_property value [gen_USERPARAMETER_ADDR_BITS_VALUE $values(C) $values(PE) $values(N)] $ADDR_BITS
}

proc validate_PARAM_VALUE.ADDR_BITS { PARAM_VALUE.ADDR_BITS } {
	# Procedure called to validate ADDR_BITS
	return true
}

proc update_PARAM_VALUE.CF { PARAM_VALUE.CF PARAM_VALUE.C PARAM_VALUE.PE } {
	# Procedure called to update CF when any of the dependent parameters in the arguments change

	set CF ${PARAM_VALUE.CF}
	set C ${PARAM_VALUE.C}
	set PE ${PARAM_VALUE.PE}
	set values(C) [get_property value $C]
	set values(PE) [get_property value $PE]
	set_property value [gen_USERPARAMETER_CF_VALUE $values(C) $values(PE)] $CF
}

proc validate_PARAM_VALUE.CF { PARAM_VALUE.CF } {
	# Procedure called to validate CF
	return true
}

proc update_PARAM_VALUE.O_BITS { PARAM_VALUE.O_BITS PARAM_VALUE.BIAS PARAM_VALUE.N } {
	# Procedure called to update O_BITS when any of the dependent parameters in the arguments change

	set O_BITS ${PARAM_VALUE.O_BITS}
	set BIAS ${PARAM_VALUE.BIAS}
	set N ${PARAM_VALUE.N}
	set values(BIAS) [get_property value $BIAS]
	set values(N) [get_property value $N]
	set_property value [gen_USERPARAMETER_O_BITS_VALUE $values(BIAS) $values(N)] $O_BITS
}

proc validate_PARAM_VALUE.O_BITS { PARAM_VALUE.O_BITS } {
	# Procedure called to validate O_BITS
	return true
}

proc update_PARAM_VALUE.BIAS { PARAM_VALUE.BIAS } {
	# Procedure called to update BIAS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BIAS { PARAM_VALUE.BIAS } {
	# Procedure called to validate BIAS
	return true
}

proc update_PARAM_VALUE.C { PARAM_VALUE.C } {
	# Procedure called to update C when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C { PARAM_VALUE.C } {
	# Procedure called to validate C
	return true
}

proc update_PARAM_VALUE.FPARG { PARAM_VALUE.FPARG } {
	# Procedure called to update FPARG when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FPARG { PARAM_VALUE.FPARG } {
	# Procedure called to validate FPARG
	return true
}

proc update_PARAM_VALUE.K { PARAM_VALUE.K } {
	# Procedure called to update K when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.K { PARAM_VALUE.K } {
	# Procedure called to validate K
	return true
}

proc update_PARAM_VALUE.N { PARAM_VALUE.N } {
	# Procedure called to update N when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.N { PARAM_VALUE.N } {
	# Procedure called to validate N
	return true
}

proc update_PARAM_VALUE.PE { PARAM_VALUE.PE } {
	# Procedure called to update PE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PE { PARAM_VALUE.PE } {
	# Procedure called to validate PE
	return true
}

proc update_PARAM_VALUE.SIGNED { PARAM_VALUE.SIGNED } {
	# Procedure called to update SIGNED when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SIGNED { PARAM_VALUE.SIGNED } {
	# Procedure called to validate SIGNED
	return true
}


proc update_MODELPARAM_VALUE.N { MODELPARAM_VALUE.N PARAM_VALUE.N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.N}] ${MODELPARAM_VALUE.N}
}

proc update_MODELPARAM_VALUE.K { MODELPARAM_VALUE.K PARAM_VALUE.K } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.K}] ${MODELPARAM_VALUE.K}
}

proc update_MODELPARAM_VALUE.C { MODELPARAM_VALUE.C PARAM_VALUE.C } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C}] ${MODELPARAM_VALUE.C}
}

proc update_MODELPARAM_VALUE.PE { MODELPARAM_VALUE.PE PARAM_VALUE.PE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PE}] ${MODELPARAM_VALUE.PE}
}

proc update_MODELPARAM_VALUE.SIGNED { MODELPARAM_VALUE.SIGNED PARAM_VALUE.SIGNED } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SIGNED}] ${MODELPARAM_VALUE.SIGNED}
}

proc update_MODELPARAM_VALUE.FPARG { MODELPARAM_VALUE.FPARG PARAM_VALUE.FPARG } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FPARG}] ${MODELPARAM_VALUE.FPARG}
}

proc update_MODELPARAM_VALUE.BIAS { MODELPARAM_VALUE.BIAS PARAM_VALUE.BIAS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BIAS}] ${MODELPARAM_VALUE.BIAS}
}

proc update_MODELPARAM_VALUE.CF { MODELPARAM_VALUE.CF PARAM_VALUE.CF } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CF}] ${MODELPARAM_VALUE.CF}
}

proc update_MODELPARAM_VALUE.ADDR_BITS { MODELPARAM_VALUE.ADDR_BITS PARAM_VALUE.ADDR_BITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ADDR_BITS}] ${MODELPARAM_VALUE.ADDR_BITS}
}

proc update_MODELPARAM_VALUE.O_BITS { MODELPARAM_VALUE.O_BITS PARAM_VALUE.O_BITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.O_BITS}] ${MODELPARAM_VALUE.O_BITS}
}
