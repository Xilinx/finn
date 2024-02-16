import os
import subprocess
from pathlib import Path

from finn.util.basic import alveo_part_map, make_build_dir

def clone_repo():
    accl_proj_dir = Path(make_build_dir(prefix="accl_proj_"))
    accl_repo_dir = accl_proj_dir / "ACCL"

    accl_repository = "https://github.com/zhenhaohe/ACCL.git"
    git_clone_accl_command = ["git", "clone", accl_repository, accl_repo_dir]
    process_git_clone = subprocess.Popen(git_clone_accl_command, stdout=subprocess.PIPE)
    process_git_clone.communicate()
    assert (
        process_git_clone.returncode == 0
    ), "Failed to clone ACCL repo. Command is: %s" % " ".join(git_clone_accl_command)

    os.chdir(accl_repo_dir)
    git_checkout_command = ["git", "checkout", "external_dma"]
    process_git_checkout = subprocess.Popen(git_checkout_command, stdout=subprocess.PIPE)
    process_git_checkout.communicate()
    assert (
        process_git_checkout.returncode == 0
    ), "Failed to checkout branch. Command is: %s" % " ".join(git_checkout_command)

    git_submodule_cmd = ["git", "submodule", "update", "--init", "--recursive"]
    process_git_submodule = subprocess.Popen(git_submodule_cmd, stdout=subprocess.PIPE)
    process_git_submodule.communicate()
    assert (
        process_git_submodule.returncode == 0
    ), "Failed to update submodules. Command is: %s" % " ".join(git_submodule_cmd)

    return str(accl_repo_dir)


def compile_internals(accl_repo_dir, fpga_part):
    # NOTE: https://support.xilinx.com/
    # s/question/0D52E00006hpYLESA2/xsct-commandline-with-no-xvfb?language=en_US
    # Somehow building the cclo requires this...
    xlsclients_path = Path("/tmp/home_dir/bin/xlsclients").expanduser()
    os.makedirs(os.path.dirname(xlsclients_path), exist_ok=True)
    with open(xlsclients_path, "w") as xlsclients:
        xlsclients.write("#!/bin/bash\n")
        xlsclients.write('echo ""\n')
    os.chmod(xlsclients_path, 0o777)

    part_to_board = {v: k for k, v in alveo_part_map.items()}
    board = part_to_board[fpga_part]

    finn_cwd = os.getcwd()

    # Now, build kernels
    os.chdir(Path(accl_repo_dir) / "test" / "refdesigns")
    part_to_board = {v: k for k, v in alveo_part_map.items()}
    board = part_to_board[fpga_part]
    coyote_board = board.lower()
    build_cclo_cmd = [
        "make",
        "-C",
        "../../kernels/cclo",
        f"PLATFORM={coyote_board}",
        "STACK_TYPE=RDMA",
        "MB_DEBUG_LEVEL=0",
        "EN_DMA=0",
        "EN_EXT_DMA=1",
    ]

    env_for_cclo = os.environ.copy()
    env_for_cclo["PATH"] = f"/tmp/home_dir/bin:{env_for_cclo['PATH']}"
    process_build_cclo = subprocess.Popen(
        build_cclo_cmd, stdout=subprocess.PIPE, env=env_for_cclo
    )
    process_build_cclo.communicate()
    assert (
        process_build_cclo.returncode == 0
    ), "Failed to build CCLO. Command is: %s" % " ".join(build_cclo_cmd)

    os.chdir(Path(accl_repo_dir) / "test" / "refdesigns")
    part_to_board = {v: k for k, v in alveo_part_map.items()}
    board = part_to_board[fpga_part]
    coyote_board = board.lower()
    build_plugins_cmd = [
        "make",
        "-C",
        "../../kernels/plugins",
        f"PLATFORM={coyote_board}",
        "STACK_TYPE=RDMA",
    ]
    process_build_plugins = subprocess.Popen(build_plugins_cmd, stdout=subprocess.PIPE)
    process_build_plugins.communicate()
    assert (
        process_build_plugins.returncode == 0
    ), "Failed to build plugins. Command is: %s" % " ".join(build_plugins_cmd)

    os.chdir(finn_cwd)

