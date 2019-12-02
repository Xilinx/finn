# import os
# import tempfile as tmp

# import finn.custom_op.registry as registry
from finn.transformation import Transformation


class Compilation(Transformation):
    """Compilation for all nodes in model"""

    def get_includes(self, node):
        # step by step addition of include paths to ensure easy extension
        include_paths = []
        include_paths.append("/workspace/finn/src/finn/data/cpp")
        include_paths.append("/workspace/cnpy/")
        include_paths.append("/workspace/finn-hlslib")
        include_paths.append("/workspace/vivado-hlslib")

    def apply(self, model):
        for node in model.graph.node:
            includes = self.get_includes(node)
        # bash_compile = """g++ -o {}/execute_{} {}/execute_{}.cpp
        # /workspace/cnpy/cnpy.cpp -I/workspace/finn/src/finn/data/cpp -I/workspace/cnpy/
        # -I/workspace/finn-hlslib -I/workspace/vivado-hlslib
        # --std=c++11 -lz""".format(
        #    self.tmp_dir, node.op_type, self.tmp_dir, node.op_type
        # )
        # process_compile = subprocess.Popen(bash_compile.split(), stdout=subprocess.PIPE)
        # process_compile.communicate()
        return (model, False)
