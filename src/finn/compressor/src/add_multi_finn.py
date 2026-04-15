"""
Generate a compressor core for FINN's add_multi module (COMP path).

The add_multi module in mvu.sv reduces N unsigned partial sums of ARG_WIDTH
bits into a single result (N dsp lanes outputs).  This script generates a LUT-mapped compressor tree
for a specific (N, ARG_WIDTH) configuration, producing a module that can be
matched by the CATCH_COMP macro in add_multi.sv.

Unlike dotp_finn.py, no absorption is needed:
  - No gates:       inputs are complete values, not partial-product factor pairs
  - No constants:   no Baugh-Wooley sign-correction (inputs are unsigned)
  - No accumulation: accumulation stays downstream in mvu.sv

Two call modes:

  Direct mode — caller supplies N and ARG_WIDTH explicitly:
    python add_multi_finn.py --n 32 --arg_width 6 -t Versal -o gen/

  MVU mode — caller supplies MVU-level parameters, and the script computes
  the required lo_width values per DSP lane via a Python replica of
  mvu.sv::sliceLanes(), then generates one compressor per unique (N, lo_width):
    python add_multi_finn.py --mvu --n 8 --version 2 --ww 2 --aw 2 \
        --accu_width 16 --narrow_weights 0 -t Versal -o gen/


Outputs:
  comp_<N>u<W>_d<delay>.sv  — the generated compressor core(s)
"""

import os
import math
import argparse
import shutil

from .main import generate_compressor
from .target import resolve_target, resolve_target_name, Versal, SevenSeries
from .utils.shape import Shape


# ---------------------------------------------------------------------------
# Python replica of mvu.sv::sliceLanes()
#
# This must mirror the SV implementation exactly. Any change to sliceLanes()
# in mvu.sv requires updating this function as well. The $warning guard in
# add_multi.sv catches divergence at simulation time.
#
# This outsourced computation is required as lane width is relevant to the
# compressor input Shape and thus needs to be known at generation time.

def clog2(n):
    """Ceiling of log2, matching SystemVerilog $clog2 semantics."""
    if n <= 1:
        return 0
    return math.ceil(math.log2(n))


def slice_lanes(version, ww, aw, accu_width, narrow_weights):
    """
    Compute DSP lane offsets — Python replica of mvu.sv::sliceLanes().
    Parameters
    ----------
    version : int
        DSP version (1=DSP48E1, 2=DSP48E2, 3=DSP58).
    ww : int
        WEIGHT_WIDTH.
    aw : int
        ACTIVATION_WIDTH.
    accu_width : int
        ACCU_WIDTH.
    narrow_weights : bool
        NARROW_WEIGHTS flag.

    Returns
    -------
    (num_lanes, offsets) : tuple
        num_lanes : int 
            number of DSP lanes.
        offsets   : list[int] 
            lane boundary positions (length num_lanes+1).
    """
    a_width = 25 + 2 * (version > 1)
    p_width = 58 if version == 3 else 48
    min_lane_width = ww + aw - 1

    if a_width == ww:
        num_lanes = 1
    else:
        num_lanes = 1 + (a_width - (0 if narrow_weights else 1) - ww) // min_lane_width

    # Distribute slack bits preferring right lanes
    bit_slack = a_width - (0 if narrow_weights else 1) - ww - (num_lanes - 1) * min_lane_width

    offsets = [0] * (num_lanes + 1)
    for i in range(1, num_lanes):
        extra = (bit_slack + (num_lanes - 1 - i)) // (num_lanes - i)
        offsets[i] = offsets[i - 1] + min_lane_width + extra
        bit_slack -= extra

    # Last lane bounded by min(ACCU_WIDTH, P_WIDTH)
    offsets[num_lanes] = offsets[num_lanes - 1] + accu_width
    if offsets[num_lanes] > p_width:
        offsets[num_lanes] = p_width

    return num_lanes, offsets


def lo_widths_from_mvu_params(version, ww, aw, accu_width, narrow_weights):
    """
    Compute the lo_width for each DSP lane.

    Returns
    -------
    list[int] 
        lo_width for lane 0 .. num_lanes-1.
    """
    num_lanes, offsets = slice_lanes(version, ww, aw, accu_width, narrow_weights)
    return [offsets[i + 1] - offsets[i] for i in range(num_lanes)]


def comp_module_name(n, arg_width, delay):
    """
    Return the compressor module name, e.g. 'comp_32u6_d4'.

    Encodes:
      N         — number of unsigned addends (= SIMD)
      ARG_WIDTH — bits per addend (= lo_width from mvu.sv lane slicing)
      delay     — pipeline stages produced by the generator

    The 'u' indicates unsigned, matching the mvu_bench naming convention.
    The delay suffix lets the CATCH_COMP macro in add_multi.sv match on
    minimum pipeline depth (DEPTH >= d).
    """
    return f"comp_{n}u{arg_width}_d{delay}"


def generate_add_multi_comp(target, n, arg_width, pipeline_every, output_dir,
                            name=None):
    """
    Generate a multi-input adder compressor (no accumulation).

    Parameters
    ----------
    target : Target
        FPGA target (Versal, SevenSeries) — selects LUT primitives.
    n : int
        Number of unsigned addends.
    arg_width : int
        Bit width of each addend.
    pipeline_every : int or None
        Insert pipeline registers every N combinational stages.
        None means purely combinational.
    output_dir : str
        Directory for the generated .sv file.
    name : str or None
        Module name override.  When None (default), the name is derived
        from (n, arg_width, delay) after generation.

    Returns
    -------
    (name, path, delay) : tuple
        Module name, file path, and pipeline depth of the generated compressor.
    """
    # Shape: W columns each of height N.
    # Each of the N operands contributes 1 bit to each of the W bit-positions,
    # so every column has the same height N.
    shape = Shape([n] * arg_width)

    # First pass: generate with a temporary name to discover the actual delay.
    # The delay depends on the compressor structure and pipeline_every, so we
    # can't know it before generation.
    tmp_name = name if name is not None else f"comp_{n}u{arg_width}"
    tmp_path = os.path.join(output_dir, tmp_name + ".sv")

    delay = generate_compressor(
        target=target,
        shape=shape,
        name=tmp_name,
        comb_depth=pipeline_every,
        accumulate=False,          # Pure adder, no fused accumulation
        accumulator_width=None,    # Not applicable without accumulation
        gates=[],                  # No gate absorption, inputs are complete values
        constants=[],              # No Baugh-Wooley correction, unsigned inputs
        path=tmp_path,
        test=False,
        enable=False,              # No accumulator registers to initialize
    )

    # Derive final name with delay suffix
    if name is not None:
        final_name = name
        final_path = tmp_path
    else:
        final_name = comp_module_name(n, arg_width, delay)
        final_path = os.path.join(output_dir, final_name + ".sv")

        if final_name != tmp_name:
            # Rename file and replace module name inside it
            with open(tmp_path, "r") as f:
                content = f.read()
            content = content.replace(tmp_name, final_name)
            with open(final_path, "w") as f:
                f.write(content)
            os.remove(tmp_path)

    return final_name, final_path, delay


def generate_add_multi_comps(fpgapart, version, simd, ww, aw, accu_width,
                             narrow_weights, output_dir):
    """
    Generate add_multi compressor cores and patch add_multi.sv.
    This is the high-level entry point called by FINN's generate_hdl().

    ALWAYS generates add_multi.sv in output_dir, either:
    - Patched version with CATCH_COMP entries if compressors are eligible
    - Copy of template if ineligible (SIMD < 4 or version == 2)

    This ensures every node has code_gen_dir/add_multi.sv, eliminating
    conditional logic in file management.

    Parameters
    ----------
    fpgapart : str
        FPGA part string.
    version : int
        DSP version (1=DSP48E1, 2=DSP48E2, 3=DSP58).
    simd, ww, aw, accu_width : int
        MVU parameters.
    narrow_weights : int
        NARROW_WEIGHTS flag (0 or 1).
    output_dir : str
        Directory for generated files (= code_gen_dir).

    Returns
    -------
    dict with keys:
        comp_names : list[str] — generated module names (empty if ineligible)
        files      : list[str] — paths of all generated/patched files
    """

    rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/")
    patched_path = os.path.join(output_dir, "add_multi.sv")

    # Check eligibility (same logic as _is_add_multi_comp_eligible)
    if version == 2 or simd < 4:
        # Ineligible: just copy template as-is
        shutil.copy(os.path.join(rtllib_dir, "add_multi.sv"), patched_path)
        return {"comp_names": [], "files": [patched_path]}

    # Eligible: generate compressors and patch add_multi.sv
    target = resolve_target(fpgapart)

    # This is currently a parallel implementation of the lo_width computation in mvu.sv's sliceLanes() function.
    # The resulting lo_width values determine the compressor input Shapes, so we need to compute them here in Python at generation time.
    # Must be kept in SYNC.
    widths = lo_widths_from_mvu_params(version, ww, aw, accu_width, narrow_weights)

    # Generate one compressor per unique (SIMD, lo_width)
    generated = {}  # (simd, width) -> (name, delay)
    for w in widths:
        key = (simd, w)
        if key not in generated:
            name, _path, delay = generate_add_multi_comp(
                target, simd, w,
                pipeline_every=1,  # Max pipelining (match dotp_comp behavior)
                output_dir=output_dir)
            generated[key] = (name, delay)

    # Copy add_multi.sv to output_dir and inject CATCH_COMP lines
    with open(os.path.join(rtllib_dir, "add_multi.sv"), "r") as f:
        add_multi_src = f.read()

    catch_lines = ""
    comp_specs = []
    for (_n, _w), (name, delay) in generated.items():
        catch_lines += "\t`CATCH_COMP(%d,%d,%d)\n" % (_n, _w, delay)
        comp_specs.append((_n, _w, delay))

    marker = "\t// FINN_GENERATED_COMP_ENTRIES\n"
    if marker not in add_multi_src:
        raise RuntimeError(
            "Cannot find FINN_GENERATED_COMP_ENTRIES marker in add_multi.sv. "
            "Has the file been modified?")
    add_multi_src = add_multi_src.replace(marker, catch_lines + marker)

    with open(patched_path, "w") as f:
        f.write(add_multi_src)

    comp_files = [os.path.join(output_dir, name + ".sv")
                  for (name, _delay) in generated.values()]

    return {
        "comp_names": [name for (name, _delay) in generated.values()],
        "comp_specs": comp_specs,  # [(N, ARG_WIDTH, DELAY), ...]
        "files": [patched_path] + comp_files,
    }


def main():
    parser = argparse.ArgumentParser(
        prog="add_multi_finn",
        description="Generate a compressor core for FINN's add_multi module."
    )
    parser.add_argument('--n', type=int, required=True,
                        help="Number of unsigned addends (= SIMD)")
    parser.add_argument('-t', '--target', default="Versal",
                        choices=["Versal", "7-Series"],
                        help="Target FPGA generation")
    parser.add_argument('-p', '--pipeline_every', type=int, default=None,
                        help="Pipeline registers every N combinational stages")
    parser.add_argument('-o', '--output_dir', default="../gen",
                        help="Output directory for generated files")
    parser.add_argument('--name', default=None,
                        help="Module name override (default: comp_<N>u<W>_d<delay>)")

    # Direct mode: explicit arg_width
    parser.add_argument('--arg_width', type=int, default=None,
                        help="Bit width per addend (direct mode)")

    # MVU mode: derive arg_width(s) from MVU parameters
    mvu_group = parser.add_argument_group(
        'MVU parameters',
        'When --mvu is given, lo_width values are computed from these '
        'MVU-level parameters (replicating mvu.sv::sliceLanes).'
    )
    mvu_group.add_argument('--mvu', action='store_true',
                           help="Enable MVU mode: derive arg_width from MVU params")
    mvu_group.add_argument('--version', type=int, default=2,
                           choices=[1, 2, 3],
                           help="DSP version (1=DSP48E1, 2=DSP48E2, 3=DSP58)")
    mvu_group.add_argument('--ww', type=int, default=None,
                           help="WEIGHT_WIDTH")
    mvu_group.add_argument('--aw', type=int, default=None,
                           help="ACTIVATION_WIDTH")
    mvu_group.add_argument('--accu_width', type=int, default=None,
                           help="ACCU_WIDTH")
    mvu_group.add_argument('--narrow_weights', type=int, default=0,
                           choices=[0, 1],
                           help="NARROW_WEIGHTS flag (0 or 1)")

    args = parser.parse_args()

    # Validate argument combinations
    if not args.mvu and args.arg_width is None:
        parser.error("Either --arg_width (direct mode) or --mvu with MVU "
                     "parameters is required.")
    if args.mvu and args.arg_width is not None:
        parser.error("--arg_width and --mvu are mutually exclusive.")
    if args.mvu:
        for param in ('ww', 'aw', 'accu_width'):
            if getattr(args, param) is None:
                parser.error(f"--mvu requires --{param}")

    target = resolve_target_name(args.target)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mvu:
        # MVU mode: compute lo_width per lane, generate unique compressors
        simd = args.n

        # For SIMD < 4, the binary adder tree is already optimal.
        # A compressor adds structural overhead with no benefit.
        if simd < 4:
            print(f"SIMD={simd} < 4: binary tree is optimal, no compressors generated.")
            return

        widths = lo_widths_from_mvu_params(
            args.version, args.ww, args.aw,
            args.accu_width, bool(args.narrow_weights)
        )
        depth = 3 + clog2(simd) + (1 if simd == 1 else 0) + 1
        add_multi_depth = depth - 4

        print(f"MVU config: VERSION={args.version} WW={args.ww} AW={args.aw} "
              f"ACCU_WIDTH={args.accu_width} NARROW_WEIGHTS={args.narrow_weights}")
        print(f"  NUM_LANES={len(widths)}  PIPELINE_DEPTH={depth}  "
              f"ADD_MULTI_DEPTH={add_multi_depth}")
        print(f"  LO_WIDTHs: {widths}")

        # Generate one compressor per unique (N, lo_width)
        seen = set()
        for lane, w in enumerate(widths):
            if (simd, w) in seen:
                print(f"  Lane {lane}: lo_width={w} — reuses existing module")
                continue
            seen.add((simd, w))

            comp_name, comp_path, comp_delay = generate_add_multi_comp(
                target, simd, w,
                args.pipeline_every, args.output_dir, name=args.name
            )
            print(f"  Lane {lane}: lo_width={w}")
            print(f"    Generated: {comp_path}")
            print(f"    Module:    {comp_name}")
            print(f"    Delay:     {comp_delay}")

    else:
        # Direct mode: single compressor for explicit arg_width
        comp_name, comp_path, comp_delay = generate_add_multi_comp(
            target, args.n, args.arg_width,
            args.pipeline_every, args.output_dir, name=args.name)

        print(f"Generated compressor core: {comp_path}")
        print(f"  Module name:     {comp_name}")
        print(f"  Configuration:   {args.n} unsigned addends x {args.arg_width} bits")
        print(f"  Pipeline depth:  {comp_delay}")


if __name__ == "__main__":
    main()
