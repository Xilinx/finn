import subprocess

import finn.core.utils as util
from finn.transformation import Transformation
from finn.core.utils import CallCppCompiler


class Compilation(Transformation):
    """Compilation for all nodes in model"""
    def __init__(self):
        super().__init__()
        self.compiler_call = CallCppCompiler()

    def get_includes(self):
        # step by step addition of include paths to ensure easy extension
        self.compiler_call.append_includes("-I/workspace/finn/src/finn/data/cpp")
        self.compiler_call.append_includes("-I/workspace/cnpy/")
        self.compiler_call.append_includes("-I/workspace/finn-hlslib")
        self.compiler_call.append_includes("-I/workspace/vivado-hlslib")
        self.compiler_call.append_includes("--std=c++11")

        
    def prepare_bash_command(self, node):
        self.get_includes()
        self.compiler_call.build(node)
        bash_command = "chmod +x " + str(self.compiler_call.compile_script)
        process_compile = subprocess.Popen( 
                bash_command.split(), stdout=subprocess.PIPE
            )   
        process_compile.communicate()
        print(self.compiler_call.code_gen_dir)

    def apply(self, model):

        for node in model.graph.node:
            if node.domain == "finn":
                backend_attribute = util.get_by_name(node.attribute, "backend")
                backend_value = backend_attribute.s.decode('UTF-8')
                if backend_value == "fpgadataflow":
                    self.prepare_bash_command(node)
                    bash_command = self.compiler_call.compile_script
                    process_compile = subprocess.Popen(
                        bash_command.split(), stdout=subprocess.PIPE
                    )
                    process_compile.communicate()

                    model.set_attribute(node, "executable_path", self.compiler_call.executable_path)

        return (model, False)
