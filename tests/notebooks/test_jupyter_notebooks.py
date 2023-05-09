import pytest

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from finn.util.basic import get_finn_root

notebook_timeout_seconds = 3600
notebook_basic_dir = get_finn_root() + "/notebooks/basics/"
notebook_advanced_dir = get_finn_root() + "/notebooks/advanced/"
notebook_cyber_dir = get_finn_root() + "/notebooks/end2end_example/cybersecurity/"
notebook_bnn_dir = get_finn_root() + "/notebooks/end2end_example/bnn-pynq/"

basics_notebooks = [
    pytest.param(notebook_basic_dir + "0_how_to_work_with_onnx.ipynb"),
    pytest.param(notebook_basic_dir + "1a_brevitas_network_import_via_FINN-ONNX.ipynb"),
    pytest.param(notebook_basic_dir + "1b_brevitas_network_import_via_QONNX.ipynb"),
]

advanced_notebooks = [
    pytest.param(notebook_advanced_dir + "0_custom_analysis_pass.ipynb"),
    pytest.param(notebook_advanced_dir + "1_custom_transformation_pass.ipynb"),
    pytest.param(notebook_advanced_dir + "2_custom_op.ipynb"),
]

cyber_notebooks = [
    pytest.param(notebook_cyber_dir + "1-train-mlp-with-brevitas.ipynb"),
    pytest.param(notebook_cyber_dir + "2-import-into-finn-and-verify.ipynb"),
    pytest.param(notebook_cyber_dir + "3-build-accelerator-with-finn.ipynb"),
]

bnn_notebooks = [
    pytest.param(notebook_bnn_dir + "cnv_end2end_example.ipynb"),
    pytest.param(notebook_bnn_dir + "tfc_end2end_example.ipynb"),
    pytest.param(notebook_bnn_dir + "tfc_end2end_verification.ipynb"),
]


@pytest.mark.notebooks
@pytest.mark.parametrize(
    "notebook", basics_notebooks + advanced_notebooks + cyber_notebooks + bnn_notebooks
)
def test_notebook_exec(notebook):
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(
            timeout=notebook_timeout_seconds, kernel_name="python3"
        )
        try:
            assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
        except Exception:
            assert False, f"Failed executing {notebook}"
