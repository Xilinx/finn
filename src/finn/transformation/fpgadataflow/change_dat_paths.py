import os
from qonnx.transformation.base import Transformation

from finn.util.context import Context

from pathlib import Path
import re


class ChangeDATPaths(Transformation):
    """Convert DAT file paths between being relative to the output directory
       and absolute paths."""

    def __init__(self, ctx: Context, abs: bool = False):
        super().__init__()
        self.ctx = ctx
        self.abs = abs
        self.rtl_suffixes = [".v", ".sv", ".vhd", ".vh"]

    def apply(self, model):
        for node in model.graph.node:

            # find the IP gen dir
            ipgen_path = self.ctx.get_subcontext(Path(node.name)).directory

            if ipgen_path is not None and os.path.isdir(ipgen_path):
                for dname, dirs, files in os.walk(ipgen_path):
                    for fname in files:
                        if any([fname.endswith(suffix) for suffix in self.rtl_suffixes]):

                            fpath = os.path.join(dname, fname)
                            with open(fpath, 'r') as f:
                                s = f.read()

                            # Regular expression to find paths ending with .dat enclosed in quotes
                            pattern = r'["\'](.*?\.dat)["\']'
                            paths = re.findall(pattern, s)

                            # Change paths between relative and absolute
                            changed_paths = []
                            for path in paths:
                                path_obj = Path(path)
                                if self.abs:
                                    # Convert paths to absolute, assume they are currently relative to output dir or containing dir.
                                    if not path_obj.is_absolute():
                                        if (Path(dname) / path_obj).is_file():
                                            path_obj = (Path(dname) / path_obj).resolve()
                                        elif (self.ctx.directory.resolve() / path_obj).is_file():
                                            path_obj = (self.ctx.directory.resolve() / path_obj)
                                        else:
                                            raise RuntimeError(f"Path {path_obj} did not exist in {dname} or {self.ctx.directory.resolve()}.")
                                else:
                                    # Convert paths to relative, assume they are currently absolute.
                                    if path_obj.is_absolute():
                                        path_obj = path_obj.relative_to(self.ctx.directory.resolve())
                                    else:
                                        raise RuntimeError(f"Path {path_obj} in {dname}/{fname} is not absolute.")
                                changed_paths.append(str(path_obj))

                            # Replace original paths with changed paths in the contents
                            for original, changed in zip(paths, changed_paths):
                                s = s.replace(original, changed)

                            with open(fpath, 'w') as f:
                                f.write(s)

        return (model, False)
