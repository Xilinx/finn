from pathlib import Path
from typing import Set, Dict, Tuple


class ContextLibraryNotFound(Exception):
    """Context raises this error if a library name could not be resolved into a path."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message

class Context:
    """Helper for builders."""

    def __init__(self, directory: Path, libraries: Dict[str,Path], fpga_part: str, clk_ns: int, clk_hls: int, top_ctx: "Context" = None, ip_name: str="finn_design", vitis: bool=False, signature=[]):
        self._directory: Path = directory            # Own directory, absolute path.
        self._libraries: Dict[str,Path] = libraries  # Known libraries, dict of {library name : path to library root}.
        self._shared: Set[Tuple[str,Path]] = set()   # Shared files, set of (library name, path relative to library root)
        self._kernel_files: Dict[str,Set[Path]] = {} # Kernel files, dict of {kernel name : {paths relative to finn root}}
        self._children: Dict[Path,Context] = {}
        self.fpga_part: str = fpga_part
        self.clk_hls: int = clk_hls
        self.clk_ns: int = clk_ns
        self.ip_name: str = ip_name
        self.vitis:bool = vitis
        self.signature = signature
        self.top_ctx: Context = top_ctx

        Path(self._directory).mkdir(exist_ok=True, parents=False)

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def shared(self) -> Set[Path]: # get own shared and all shared from below
        shared_all: Set[Path] = set()
        for lib, path in self._shared:
            try:
                shared_all.add(self._libraries[lib] / path)
            except KeyError:
                raise ContextLibraryNotFound(f"Library {lib} not found in {self._libraries}, for dependency {path} in kernel {self._directory}.")

        for _, subctx in self._children.items():
            shared_all = shared_all | subctx.shared

        return shared_all

    @property
    def kernel_files(self) -> Dict[Path,Set[Path]]: # get own kernel files and all kernel files from below
        kernel_files_all: Dict[Path,Set[Path]] = {}
        for kernel_name, paths in self._kernel_files.items():
            kernel_path = Path(kernel_name)
            kernel_files_all[kernel_path] = set()
            for path in paths:
                kernel_files_all[kernel_path].add(path)

        for _, subctx in self._children.items():
            for kernel_path, paths in subctx.kernel_files.items():
                if kernel_path in kernel_files_all.keys():
                    kernel_files_all[kernel_path] = kernel_files_all[kernel_path] | paths
                else:
                    kernel_files_all[kernel_path] = paths

        return kernel_files_all

    @property
    def children(self) -> Dict[str,"Context"]:
        return self._children

    def add_shared(self, sharedFile: Tuple[str,Path]) -> None:
        self._shared.add(sharedFile)

    def add_kernel_file(self, kernel_name: str, kernelFile: Path) -> None:
        if kernel_name in self._kernel_files.keys():
            self._kernel_files[kernel_name].add(kernelFile)
        else:
            self._kernel_files[kernel_name] = set({kernelFile})

    def get_subcontext(self, subdirectory: Path) -> "Context":
        if subdirectory in self._children.keys():
            return self._children[subdirectory]
        else:
            top_ctx = self if self.top_ctx == None else self.top_ctx
            child = Context(
                directory=self.directory / subdirectory,
                libraries=self._libraries,
                fpga_part=self.fpga_part,
                clk_ns=self.clk_ns,
                clk_hls=self.clk_hls,
                top_ctx=top_ctx,
                ip_name=self.ip_name,
                vitis=self.vitis,
                signature=self.signature)
            self._children[subdirectory] = child
            return child

    @property
    def shared_dir(self) -> Path: # Return path to shared RTL folder at top level dir
        top_ctx = self if self.top_ctx == None else self.top_ctx
        shared_dir_path = top_ctx.directory / Path('shared')
        shared_dir_path.mkdir(exist_ok=True)
        return shared_dir_path

    def get_kernel_dir(self, kernel_name: str) -> Path: # Return path to kernel file folder at top level dir
        top_ctx = self if self.top_ctx == None else self.top_ctx
        kernel_dir_path = top_ctx.directory / Path(kernel_name)
        kernel_dir_path.mkdir(exist_ok=True)
        return kernel_dir_path

    def resolve_library(self, shared_dep: Tuple[str,Path]):
        lib, path = shared_dep
        try:
            return self._libraries[lib] / path
        except KeyError:
            raise ContextLibraryNotFound(f"Library {lib} not found in {self._libraries}, for dependency {path}.")
