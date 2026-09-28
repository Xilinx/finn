# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import shutil
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.util.fpgadataflow import is_hls_node, is_rtl_node


class ExportPortableRTL(Transformation):
    """Export a self-contained, portable RTL project from a stitched FINN design.

    This transformation copies all Verilog source files and memory initialization
    files (.dat) to a single directory structure with relative paths, suitable for
    use in external simulators (QuestaSim, ModelSim, Verilator) or non-IPI Vivado
    projects.

    Parameters
    ----------
    export_dir : str
        Target directory for the portable RTL export
    """

    def __init__(self, export_dir):
        super().__init__()
        self.export_dir = export_dir

    def apply(self, model):
        # Verify stitched IP exists
        vivado_stitch_proj = model.get_metadata_prop("vivado_stitch_proj")
        assert vivado_stitch_proj, "No stitched IP found. Run CreateStitchedIP first."
        assert os.path.isdir(
            vivado_stitch_proj
        ), f"Stitched IP directory not found: {vivado_stitch_proj}"

        # Create export directory structure
        rtl_dir = os.path.join(self.export_dir, "rtl")
        src_dir = os.path.join(rtl_dir, "src")
        rtllib_dir = os.path.join(rtl_dir, "rtllib")
        data_dir = os.path.join(self.export_dir, "data")

        for d in [src_dir, rtllib_dir, data_dir]:
            os.makedirs(d, exist_ok=True)

        # Track all files for file list generation
        all_rtl_files = []
        rtllib_files_copied = set()

        # 1. Copy all RTL files from all_verilog_srcs.txt
        all_verilog_srcs_file = os.path.join(vivado_stitch_proj, "all_verilog_srcs.txt")
        assert os.path.isfile(
            all_verilog_srcs_file
        ), f"all_verilog_srcs.txt not found in {vivado_stitch_proj}"

        with open(all_verilog_srcs_file, "r") as f:
            verilog_files = f.read().strip().split()

        for src_file in verilog_files:
            if not os.path.isfile(src_file):
                continue
            # Skip XCI files (Vivado-specific IP)
            if src_file.endswith(".xci"):
                continue

            basename = os.path.basename(src_file)

            if "finn-rtllib" in src_file:
                # Copy finn-rtllib files (preserving subdirectory structure)
                if src_file not in rtllib_files_copied:
                    rtllib_rel = src_file.split("finn-rtllib/")[1]
                    rtllib_subdir = os.path.dirname(rtllib_rel)
                    if rtllib_subdir:
                        dest_subdir = os.path.join(rtllib_dir, rtllib_subdir)
                        os.makedirs(dest_subdir, exist_ok=True)
                    dest_file = os.path.join(rtllib_dir, rtllib_rel)
                    shutil.copy2(src_file, dest_file)
                    all_rtl_files.append(f"rtl/rtllib/{rtllib_rel}")
                    rtllib_files_copied.add(src_file)
            else:
                # All other files go to src directory
                dest_file = os.path.join(src_dir, basename)
                if not os.path.exists(dest_file):
                    shutil.copy2(src_file, dest_file)
                    all_rtl_files.append(f"rtl/src/{basename}")

        # 2. Collect .dat files from nodes and scan src files for additional references
        dat_mapping = {}  # old_abs_path -> new_rel_path (relative to src_dir)

        # Collect .dat files from all nodes
        for node in model.graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue

            node_inst = getCustomOp(node)
            dat_files = self._get_node_dat_files(node_inst)

            for src_dat in dat_files:
                if os.path.isfile(src_dat) and src_dat not in dat_mapping:
                    dat_basename = os.path.basename(src_dat)
                    dest_dat = os.path.join(data_dir, dat_basename)
                    if not os.path.exists(dest_dat):
                        shutil.copy2(src_dat, dest_dat)
                    # Relative path from src_dir to data_dir
                    rel_path = os.path.relpath(dest_dat, src_dir)
                    dat_mapping[src_dat] = rel_path

        # Scan src files for any additional $readmemh references
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                if f.endswith((".v", ".sv")):
                    fpath = os.path.join(root, f)
                    with open(fpath, "r") as file:
                        content = file.read()

                    pattern = r'\$readmemh\s*\(\s*"([^"]+)"'
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if os.path.isabs(match) and os.path.isfile(match):
                            if match not in dat_mapping:
                                dat_basename = os.path.basename(match)
                                dest_dat = os.path.join(data_dir, dat_basename)
                                if not os.path.exists(dest_dat):
                                    shutil.copy2(match, dest_dat)
                                rel_path = os.path.relpath(dest_dat, src_dir)
                                dat_mapping[match] = rel_path

        # 3. Rewrite absolute paths in all src Verilog files
        self._rewrite_verilog_paths(src_dir, dat_mapping)

        # 4. Generate file lists
        self._generate_filelist_f(all_rtl_files)
        self._generate_sources_tcl(all_rtl_files)
        self._generate_readme()

        return (model, False)

    def _get_node_dat_files(self, node_inst):
        """Get all .dat memory initialization files for a node."""
        dat_files = []

        # Try get_all_meminit_filenames if available (some RTL nodes)
        if hasattr(node_inst, "get_all_meminit_filenames"):
            try:
                dat_files = node_inst.get_all_meminit_filenames(abspath=True)
            except Exception:
                pass

        # Also scan code_gen_dir_ipgen for .dat files
        try:
            ipgen_path = node_inst.get_nodeattr("code_gen_dir_ipgen")
            if ipgen_path and os.path.isdir(ipgen_path):
                for root, dirs, files in os.walk(ipgen_path):
                    for f in files:
                        if f.endswith(".dat"):
                            full_path = os.path.join(root, f)
                            if full_path not in dat_files:
                                dat_files.append(full_path)
        except Exception:
            pass

        return dat_files

    def _rewrite_verilog_paths(self, rtl_dir, path_mapping):
        """Rewrite absolute $readmemh paths to relative paths in Verilog files."""
        for root, dirs, files in os.walk(rtl_dir):
            for f in files:
                if f.endswith((".v", ".sv")):
                    fpath = os.path.join(root, f)
                    with open(fpath, "r") as file:
                        content = file.read()

                    modified = False

                    # Replace absolute paths with relative paths
                    for old_path, new_path in path_mapping.items():
                        if old_path in content:
                            content = content.replace(old_path, new_path)
                            modified = True

                    # Handle any remaining absolute paths by basename matching
                    pattern = r'\$readmemh\s*\(\s*"([^"]+)"'
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if os.path.isabs(match):
                            basename = os.path.basename(match)
                            for old, new in path_mapping.items():
                                if os.path.basename(old) == basename:
                                    content = content.replace(match, new)
                                    modified = True
                                    break

                    if modified:
                        with open(fpath, "w") as file:
                            file.write(content)

    def _generate_filelist_f(self, rtl_files):
        """Generate filelist.f for Verilator/QuestaSim/ModelSim."""
        filelist_path = os.path.join(self.export_dir, "filelist.f")
        with open(filelist_path, "w") as f:
            f.write("// FINN Portable RTL Export - File List\n")
            f.write("// For use with Verilator, QuestaSim, ModelSim\n")
            f.write("// Usage: verilator -f filelist.f --top-module finn_design_wrapper\n")
            f.write("//        vlog -f filelist.f\n\n")

            # Add include directories
            f.write("// Include directories\n")
            f.write("+incdir+rtl/rtllib\n")
            f.write("+incdir+rtl/src\n\n")

            # Add source files (rtllib first, then src)
            f.write("// RTL library files\n")
            for rtl_file in sorted(rtl_files):
                if "rtllib" in rtl_file:
                    f.write(f"{rtl_file}\n")

            f.write("\n// Design files\n")
            for rtl_file in sorted(rtl_files):
                if "src" in rtl_file:
                    f.write(f"{rtl_file}\n")

    def _generate_sources_tcl(self, rtl_files):
        """Generate sources.tcl for Vivado non-IPI projects."""
        tcl_path = os.path.join(self.export_dir, "sources.tcl")
        with open(tcl_path, "w") as f:
            f.write("# FINN Portable RTL Export - Vivado Sources\n")
            f.write("# For use with non-IPI Vivado projects\n")
            f.write("# Usage: source sources.tcl\n\n")

            f.write("# Get the directory containing this script\n")
            f.write("set script_dir [file dirname [info script]]\n\n")

            f.write("# Add all RTL source files\n")
            for rtl_file in sorted(rtl_files):
                f.write(f'add_files -norecurse "${{script_dir}}/{rtl_file}"\n')

            f.write("\n# Set include directories\n")
            f.write("set_property include_dirs [list \\\n")
            f.write('    "${script_dir}/rtl/rtllib" \\\n')
            f.write('    "${script_dir}/rtl/src" \\\n')
            f.write("] [current_fileset]\n")

    def _generate_readme(self):
        """Generate README with usage instructions."""
        readme_path = os.path.join(self.export_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write("# FINN Portable RTL Export\n\n")
            f.write("This directory contains a self-contained RTL export of a FINN design.\n\n")
            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write("portable_rtl/\n")
            f.write("├── rtl/\n")
            f.write("│   ├── src/        # Design source files\n")
            f.write("│   └── rtllib/     # Shared FINN RTL library modules\n")
            f.write("├── data/           # Memory initialization files (.dat)\n")
            f.write("├── filelist.f      # File list for simulators\n")
            f.write("├── sources.tcl     # Vivado source script\n")
            f.write("└── README.md       # This file\n")
            f.write("```\n\n")
            f.write("## Usage\n\n")
            f.write("### Verilator\n")
            f.write("```bash\n")
            f.write("verilator -f filelist.f --top-module finn_design_wrapper\n")
            f.write("```\n\n")
            f.write("### QuestaSim / ModelSim\n")
            f.write("```bash\n")
            f.write("vlog -f filelist.f\n")
            f.write("vsim finn_design_wrapper\n")
            f.write("```\n\n")
            f.write("### Vivado (non-IPI)\n")
            f.write("```tcl\n")
            f.write("source sources.tcl\n")
            f.write("set_property top finn_design_wrapper [current_fileset]\n")
            f.write("```\n\n")
            f.write("## Notes\n\n")
            f.write("- All `$readmemh` paths have been converted to relative paths\n")
            f.write("- Memory initialization files (.dat) are in the data/ directory\n")
            f.write("- The design uses FINN's streaming dataflow architecture\n")
