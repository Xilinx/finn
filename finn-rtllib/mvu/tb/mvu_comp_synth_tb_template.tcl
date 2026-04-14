# Vivado synthesis flow for MVU compressor integration.
# Runs synthesis (no simulation) to verify area and timing.
#
# Template placeholders expanded by run_mvu_comp_synth_tests.sh:
#   {label}       - Configuration label
#   {mvu_dir}     - Absolute path to finn/finn-rtllib/mvu/
#   {comp_dir}    - Absolute path to compressor source
#   {gen_dir}     - Absolute path to gen/<label>/
#   {mh}          - Matrix Height
#   {mw}          - Matrix Width
#   {pe}          - Processing Elements
#   {simd}        - SIMD lanes
#   {ww}          - Weight Width
#   {aw}          - Activation Width
#   {accu_width}  - Accumulator Width
#   {signed_act}  - Signed Activations (0 or 1)
#   {comp_depth}  - Compressor Pipeline Depth

set label {label}
set part xcvc1902-vsva2197-2MP-e-S

# Design sources (non-project mode)
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

# Run Synthesis (non-project mode — direct synth_design with -generic)
synth_design -top mvu_vvu_axi -part $part -generic [join { \
	IS_MVU=1 \
	VERSION=3 \
	MW={mw} \
	MH={mh} \
	PE={pe} \
	SIMD={simd} \
	ACTIVATION_WIDTH={aw} \
	WEIGHT_WIDTH={ww} \
	ACCU_WIDTH={accu_width} \
	SIGNED_ACTIVATIONS={signed_act} \
	USE_COMPRESSOR=1 \
	COMP_PIPELINE_DEPTH={comp_depth} \
}]

# Report utilization
report_utilization -file {gen_dir}/mvu_comp_synth_{label}_util.rpt
report_timing_summary -file {gen_dir}/mvu_comp_synth_{label}_timing.rpt

quit
