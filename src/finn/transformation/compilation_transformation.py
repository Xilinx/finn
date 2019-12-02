import subprocess

import finn.core.utils as util
from finn.transformation import Transformation


class Compilation(Transformation):
    """Compilation for all nodes in model"""

    def get_includes(self, node):
        # step by step addition of include paths to ensure easy extension
        include_paths = []
        include_paths.append("-I/workspace/finn/src/finn/data/cpp")
        include_paths.append("-I/workspace/cnpy/")
        include_paths.append("-I/workspace/finn-hlslib")
        include_paths.append("-I/workspace/vivado-hlslib")
        include_paths.append("--std=c++11")

        return include_paths

    def prepare_bash_command(self, node, code_gen_dir):
        cpp_files = []
        cpp_files.append(str(code_gen_dir) + "/execute_" + str(node.op_type) + ".cpp")
        includes = self.get_includes(node)
        for lib in includes:
            if "cnpy" in lib:
                cpp_files.append("/workspace/cnpy/cnpy.cpp")
                includes.append("-lz")
        executable_path = str(code_gen_dir) + "/execute_" + str(node.op_type)

        compile_components = []
        compile_components.append("g++ -o " + str(executable_path))
        for cpp_file in cpp_files:
            compile_components.append(cpp_file)
        for lib in includes:
            compile_components.append(lib)

        bash_compile = ""
        for component in compile_components:
            bash_compile += str(component) + " "

        return bash_compile

    def apply(self, model):

        for node in model.graph.node:
            code_gen_dir = util.get_by_name(node.attribute, "code_gen_dir")
            code_gen_dir = code_gen_dir.s.decode("UTF-8")
            if not code_gen_dir:
                raise ValueError(
                    """There is no generated code to compile
                        for node of op type {}""".format(
                        node.op_type
                    )
                )
            else:
                bash_command = self.prepare_bash_command(node, code_gen_dir)
                process_compile = subprocess.Popen(
                    bash_command.split(), stdout=subprocess.PIPE
                )
                process_compile.communicate()

                executable_path = str(code_gen_dir) + "/execute_" + str(node.op_type)
                model.set_attribute(node, "executable_path", executable_path)

        return (model, False)
