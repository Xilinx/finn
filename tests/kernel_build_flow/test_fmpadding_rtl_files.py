from finn.util.context import Context
from pathlib import Path
from finn.kernels.fmpadding.rtl.fmpadding_rtl import FMPaddingRTL
import importlib.resources
import tempfile


def test_fmpadding_files():
    """ This test checks that FMPaddingRTL can generate a toplevel
        and that its shared files are present """

    # Make an instance of kernel.
    k = FMPaddingRTL(name='TestFMPad', ImgDim=[4,4], Padding=[1,1,1,1], NumChannels=3,
                     SIMD=4, inputDataType="INT16", numInputVectors=1, dynamic_mode=False)

    with tempfile.TemporaryDirectory() as temp_dir:

        ctx = Context(Path(temp_dir), libraries={}, fpga_part="", clk_ns=5, clk_hls=5)

        k.generate_instance_files(ctx)
  
        try:
            toplevel_path = Path(temp_dir) / "TestFMPad.v"
        except:
            raise RuntimeError("Top level file missing or using unexpected filename.")

        with open(toplevel_path, 'r') as f:
            contents = f.read()

    # Check that top level generates correctly.
    if ("fmpadding_axi" not in contents) or ("TestFMPad" not in contents):
        raise RuntimeError(f"FMPaddingRTL top-level does not contain module name or instance name.")

    # Check if all 3 shared files and only 3 shared files are found.
    shared = k.kernelFiles

    resource_dir = importlib.resources.files("finn") / list(shared)[0]
    files_content = {}  
    # Iterate over all items in the directory  
    for file in resource_dir.iterdir():  
        if file.is_file():  
            # Read the file's content as a string  
            content = file.read_text(encoding='utf-8')  
            files_content[file.name] = content
    
    if len(files_content.items()) != 3:
        raise RuntimeError(f"Did not find FMPaddingRTL shared files.")
