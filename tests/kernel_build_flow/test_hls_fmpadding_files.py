import tempfile
from finn.kernels import gkr
from finn.util.context import Context
from pathlib import Path
import importlib
import os


def test_fmpadding_filegen()->None:
    """ A test that tries to create a temporary directory that has
    the shared files that FMPadding requires along with the generated
    files. 
    
    This is probably just a temporary test while the Context work is being built up.

    """

    config = {
        "ImgDim" : (128,384),
        "Padding" : (1,1,1,1),
        "NumChannels" : 4,
        "SIMD" : 1,
        "inputDataType": "INT16",
        "dynamic_mode": 0,
        "numInputVectors" : 4,
        "name" : "FMPaddingTestInst"
    }

    check = [f'{config["name"]}.cpp']

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        k = gkr.kernel("FMPadding", config)

        ctx = Context(temp_path, libraries={"finn-hlslib" : Path(os.environ["FINN_ROOT"]) / Path('deps/finn-hlslib')}, fpga_part="", clk_ns=5, clk_hls=5)

        # Check if includes are in shared files
        include_found = False
        for lib, path in k.sharedFiles:
            file_path = ctx.resolve_library((lib,path)) / Path("streamtools.h")
            if file_path.exists() and file_path.is_file():
                include_found = file_path
                break
        if not include_found:
            raise RuntimeError(f"streamtools.h was not found in sharedFiles of FMPaddingHLS kernel.")

        # Generate the toplevel wrapper
        k.generate_instance_files(ctx)

        output_paths = list(temp_path.iterdir())
        output_files = [x.name for x in output_paths]

        for of in check:
            if not of in output_files:
                raise RuntimeError(f"File {of} was expected in the output {output_files} of the kernel {k} but it could not be found.")
