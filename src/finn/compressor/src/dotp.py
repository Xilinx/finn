import sys, re, os
from .main import generate_compressor
from .target import Target, Versal, SevenSeries
from .utils.shape import Shape
from .utils.mul_comp_map import MulCompMap
from typing import Optional, List


if __name__ == "__main__":

	# Parse and extract Parameters from Command Line
	sig = sys.argv[1]
	_ = re.fullmatch("(\\d+)x([us])(\\d+)([us])(\\d+)", sig).groups()
	(n, na, nb, sa, sb) = (int(_[0]), int(_[2]), int(_[4]), _[1] == 's', _[3] == 's')
	assert nb <= na

	# Target platform: ca/accu goes in argv[2], target in argv[3] (default versal)
	target_arg = sys.argv[3] if len(sys.argv) > 3 else "versal"
	if target_arg == "7series":
		target = SevenSeries()
		fpga_part = "xc7z020clg400-1"
	else:  # versal (default)
		target = Versal()
		fpga_part = "xcvc1902-vsva2197-2MP-e-S"

	clog2 = lambda x: (x-1).bit_length()
	np = clog2(n) + (na if nb == 1 and not sb else na+nb) if na > 1 else (
			clog2(n+1) if sa == sb else 1 + clog2(n)
		)

	map = MulCompMap(na, nb, sa, sb)
	shape = [col * n for col in map.shape()]
	print("Shape: ", ' '.join((':'.join((f"{val:x}" for val in col)) for col in shape[::-1])))

	# Absolute Term Contribution
	constants = []
	abs_term  = n * map.absolute_term()
	# Move absolute term into absorbed constant if requested
	if len(sys.argv) > 2 and sys.argv[2] == 'ca':
		print("Constant absorption.")
		if abs_term < 0:
			abs_term += 2**np
		constants = [(abs_term >> i) & 1 for i in range(np)]
		abs_term  = 0

	name = "comp_" + sig
	# Write to gen/ relative to this script's parent directory (compressor/)
	script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	output_path = os.path.join(script_dir, "gen", name + ".sv")
	generate_compressor(
		target            = target,
		shape             = Shape((len(col) for col in shape)),
		name              = name,
		comb_depth        = None,
		accumulate        = False,
		accumulator_width = None,
		gates = [[f"{val:x}" for val in col] for col in shape],
		constants = constants,
		path = output_path,
		test = False
	)

	# Process templates with absolute paths
	gen_dir = os.path.join(script_dir, "gen")
	hdl_dir = os.path.join(script_dir, "hdl")
	for (src_rel, dst_rel) in (
		("dotp_template.sv", "dotp_"+sig+".sv"),
		("dotp_tb_template.sv", "dotp_"+sig+"_tb.sv"),
		("dotp_template.tcl", "dotp_"+sig+".tcl")
	):
		src = os.path.join(hdl_dir, src_rel)
		dst = os.path.join(gen_dir, dst_rel)
		with open(src, "rt") as fsrc:
			with open(dst, "wt") as fdst:
				for l in fsrc:
					fdst.write(l
						.replace("{n}", str(n))
						.replace("{na}", str(na))
						.replace("{nb}", str(nb))
						.replace("{sa}", 's' if sa else 'u')
						.replace("{sb}", 's' if sb else 'u')
						.replace("{signed_a}", str(int(sa)))
						.replace("{signed_b}", str(int(sb)))
						.replace("{abs_term}", str(abs_term))
						.replace("{part}", fpga_part)
						# Replace relative paths with absolute paths for TCL
						.replace("hdl/", hdl_dir + "/")
						.replace("gen/", gen_dir + "/")
					)
