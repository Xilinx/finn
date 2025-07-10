from finn.util.context import Context
from finn.util import templates
from finn.util.basic import which
from finn.kernels import Kernel
from finn.kernels.cache_manager import cache_manager
from finn.kernels.cache_config_extractor import extract_kernel_config_for_cache

from pathlib import Path
import importlib
import os
import subprocess


def gen_hls_node(kernel: Kernel, node_ctx: Context):

    # Ensure vitis_hls is available
    assert which("vitis_hls") is not None, "vitis_hls not found in PATH"

    # Extract kernel configuration for caching using standardized logic
    kernel_config = extract_kernel_config_for_cache(kernel)

    # Check if cached files exist and are valid
    cache_hash = cache_manager._compute_hash(kernel, kernel_config)
    if cache_manager.has_cached_files(kernel, kernel_config):
        print(f"Using cached files for kernel {kernel.name} (hash: {cache_hash[:12]}...)")
        if cache_manager.get_cached_files(kernel, kernel_config, node_ctx.directory):
            # Verify the cached IP directory exists
            ipgen_path = node_ctx.directory / Path(f"project_{kernel.name}")
            ip_path = ipgen_path / Path("sol1/impl/ip")
            if ip_path.exists() and ip_path.is_dir():
                print(f"Successfully restored cached IP for {kernel.name} (hash: {cache_hash[:12]}...)")
                return
            else:
                print(f"Cached files incomplete for {kernel.name} (hash: {cache_hash[:12]}...), regenerating...")
        else:
            print(f"Failed to retrieve cached files for {kernel.name} (hash: {cache_hash[:12]}...), regenerating...")

    # Generate instance HLS files in kernel subdirectory
    kernel.generate_instance_files(node_ctx)

    # Generate TCL script for ipgen
    default_directives = [
        "set_param hls.enable_hidden_option_error false",
        "config_compile -disable_unroll_code_size_check -pipeline_style flp",
        "config_interface -m_axi_addr64",
        "config_rtl -module_auto_prefix",
        "config_rtl -deadlock_detection none",
    ]
    code_gen_dict = {}
    code_gen_dict["$PROJECTNAME$"] = f"project_{kernel.name}"
    code_gen_dict["$FPGAPART$"] = node_ctx.fpga_part
    code_gen_dict["$TOPFXN$"] = kernel.name
    code_gen_dict["$CLKPERIOD$"] = str(node_ctx.clk_hls)
    code_gen_dict["$DEFAULT_DIRECTIVES$"] = "\n".join(default_directives)
    shared_includes = [node_ctx.resolve_library(shared_dep) for shared_dep in kernel.sharedFiles]
    kernel_includes = [importlib.resources.files("finn") / path for path in kernel.kernelFiles]
    code_gen_dict["$INCLUDE_SOURCES$"] = " \\\n".join(["-I"+str(path) for path in shared_includes])
    code_gen_dict["$INCLUDE_SOURCES$"] += " \\\n".join(["-I"+str(path) for path in kernel_includes])

    tcl_script_path = Path(f"hls_syn_{kernel.name}.tcl")
    template = templates.ipgentcl_template
    for key, code_gen_line in code_gen_dict.items():
        template = template.replace(key, code_gen_line)
    with open(node_ctx.directory / tcl_script_path, "w") as f:
        f.write(template)

    # Generate ipgen script
    ipgen_path = (node_ctx.directory / Path(f"project_{kernel.name}"))
    ipgen_script_path = node_ctx.directory / Path("ipgen.sh")
    with open(ipgen_script_path, "w") as f:
        f.write("#!/bin/bash \n")
        f.write("pushd {}\n".format(node_ctx.directory))
        f.write("vitis_hls %s\n" % (tcl_script_path))
        f.write("popd\n")

    # Run ipgen script
    bash_command = ["bash", ipgen_script_path]
    process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    process_compile.communicate()

    # Check if IP generated successfully
    assert os.path.isdir(ipgen_path), f"HLSCodeBuilder failed: {ipgen_path} not found."
    ip_path = ipgen_path / Path("sol1/impl/ip")
    assert os.path.isdir(ip_path), f"HLSCodeBuilder failed: {ip_path} not found. Check log under {node_ctx.directory}."

    # Store generated files in cache for future use
    if cache_manager.store_generated_files(kernel, kernel_config, node_ctx.directory):
        print(f"Cached generated files for kernel {kernel.name} (hash: {cache_hash[:12]}...)")
    else:
        print(f"Warning: Failed to cache generated files for kernel {kernel.name} (hash: {cache_hash[:12]}...)")
