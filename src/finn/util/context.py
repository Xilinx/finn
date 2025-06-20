from pathlib import Path
from typing import Set, Dict, Tuple
from dataclasses import dataclass, field


class ContextLibraryNotFound(Exception):
    """Context raises this error if a library name could not be resolved into a path."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message

@dataclass
class Context:
    """Helper for builders."""

    directory: Path           # Associated directory, absolute path or relative to cwd
    libraries: Dict[str,Path] # Known libraries, dict of {library name : path to library root}

    fpga_part : str
    clk_ns    : int
    clk_hls   : int
    ip_name   : str  = "finn_design"
    vitis     : bool = False
    signature : list = field(default_factory=list)

    rtlsim_trace:  Path                 = ""
    rtlsim_so:     Path                 = ""
    top_ctx:      "Context"             = None
    _shared:       Set[Tuple[str,Path]] = field(default_factory=set)  # Shared files, set of (library name, path relative to library root)
    _kernel_files: Dict[str,Set[Path]]  = field(default_factory=dict) # Kernel files, dict of {kernel name : {paths relative to finn root}}
    _children:     Dict[Path,"Context"] = field(default_factory=dict) # Child contexts with associated subdirectories relative to self

    def __post_init__(self):
        self.directory = Path(self.directory)
        self.directory.mkdir(exist_ok=True, parents=False)

    # Get own shared files and all shared files from lower in the hierarchy
    @property
    def shared(self) -> Set[Path]:
        shared_all: Set[Path] = set()
        for lib, path in self._shared:
            try:
                shared_all.add(self.libraries[lib] / path)
            except KeyError:
                raise ContextLibraryNotFound(f"Library {lib} not found in {self.libraries}, for dependency {path} in kernel {self.directory}.")

        for _, subctx in self._children.items():
            shared_all = shared_all | subctx.shared

        return shared_all

    # Get own kernel files and all kernel files from lower in the hierarchy
    @property
    def kernel_files(self) -> Dict[Path,Set[Path]]:
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
    def children(self) -> Dict[Path,"Context"]:
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
            return self._children[Path(subdirectory)]
        else:
            top_ctx = self if self.top_ctx == None else self.top_ctx
            child = Context(
                directory=self.directory / subdirectory,
                libraries=self.libraries,
                fpga_part=self.fpga_part,
                clk_ns=self.clk_ns,
                clk_hls=self.clk_hls,
                ip_name=self.ip_name,
                vitis=self.vitis,
                signature=self.signature,
                rtlsim_trace=self.rtlsim_trace,
                rtlsim_so=self.rtlsim_so,
                top_ctx=top_ctx)
            self._children[Path(subdirectory)] = child
            return child

    def update_subcontext(self, subctx: "Context") -> None:
        subctx.top_ctx = self if self.top_ctx == None else self.top_ctx
        for _, subchild in subctx._children.items():
            subctx.update_subcontext(subchild)
        self._children[Path(subctx.directory.name)] = subctx

    # Get path to shared RTL folder at top level dir
    @property
    def shared_dir(self) -> Path:
        top_ctx = self if self.top_ctx == None else self.top_ctx
        shared_dir_path = top_ctx.directory / Path('shared')
        shared_dir_path.mkdir(exist_ok=True)
        return shared_dir_path

    # Get path to kernel file folder at top level dir
    def get_kernel_dir(self, kernel_name: str) -> Path:
        top_ctx = self if self.top_ctx == None else self.top_ctx
        kernel_dir_path = top_ctx.directory / Path(kernel_name)
        kernel_dir_path.mkdir(exist_ok=True)
        return kernel_dir_path

    def resolve_library(self, shared_dep: Tuple[str,Path]):
        lib, path = shared_dep
        try:
            return self.libraries[lib] / path
        except KeyError:
            raise ContextLibraryNotFound(f"Library {lib} not found in {self.libraries}, for dependency {path}.")
