import os
import subprocess

from finn.transformation import Transformation


class SynthPYNQProject(Transformation):
    """Run synthesis for the PYNQ project for this graph. The MakePYNQProject
    transformation must be applied prior to this transformation."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        vivado_pynq_proj_dir = model.get_metadata_prop("vivado_pynq_proj")
        if vivado_pynq_proj_dir is None or (not os.path.isdir(vivado_pynq_proj_dir)):
            raise Exception("No synthesis project, apply MakePYNQProject first.")
        synth_project_sh = vivado_pynq_proj_dir + "/synth_project.sh"
        if not os.path.isfile(synth_project_sh):
            raise Exception("No synthesis script, apply MakePYNQProject first.")
        bash_command = ["bash", synth_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        # TODO set bitfile attribute
        # TODO pull out synthesis statistics and put them in as attributes
        return (model, False)
