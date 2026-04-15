# Vivado batch flow for MVU compressor integration test.
# Behavioral simulation — verifies the full mvu_vvu_axi pipeline with
# USE_COMPRESSOR=1 through AXI-Stream interfaces.
#
# Template placeholders expanded by run_mvu_comp_tests.sh:
#   {label}       - Configuration label (e.g. mh16_mw8_pe2_simd8_ww2_aw2_sa)
#   {mvu_dir}     - Absolute path to finn/finn-rtllib/mvu/
#   {comp_dir}    - Absolute path to compressor-python/src/ or compressor copy
#   {gen_dir}     - Absolute path to gen/<label>/

set label {label}
set tb mvu_comp_{label}_tb
set part xcvc1902-vsva2197-2MP-e-S
create_project -force mvu_comp_$label mvu_comp_$label.vivado -part $part

# Design sources:
#   MVU pipeline:  mvu_pkg.sv, mvu_vvu_axi.sv, replay_buffer.sv,
#                  mvu_vvu_8sx9_dsp58.sv, mvu.sv, add_multi.sv
#   Compressor:    dotp_comp.sv (expanded), mul_comp_map.sv, comp_<sig>.sv
read_verilog -sv \
	{mvu_dir}/mvu_pkg.sv \
	{mvu_dir}/mvu_vvu_axi.sv \
	{mvu_dir}/replay_buffer.sv \
	{mvu_dir}/mvu_vvu_8sx9_dsp58.sv \
	{mvu_dir}/mvu.sv \
	{mvu_dir}/add_multi.sv \
	{comp_dir}/finn/compressor/hdl/mul_comp_map.sv \
	{gen_dir}/dotp_comp.sv \
	{*}[glob {gen_dir}/comp_*.sv]

# Simulation sources
set simset [current_fileset -simset]
add_files -fileset $simset {gen_dir}/$tb.sv
set_property top $tb $simset
set_property xsim.simulate.runtime all $simset

# Defines for simulation-only features
set_property verilog_define {FINN_SIMULATION=1} $simset

# Run Simulation
launch_simulation
close_sim

quit
