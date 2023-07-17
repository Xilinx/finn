import os
import numpy as np
from scipy.stats import linregress
import subprocess
import pytest
import itertools
import logging

# no __init__ constructors allowed in Pytest - so use global variables instead
base_dir_global = os.getcwd()
default_test_run_timeout = 30 # seconds
output_execute_results_file = "output.npy"
execute_results_reference_file = "output_reference.npy"
output_throughput_results_file = "nw_metrics.txt"
throughput_results_formatted_file = "throughput_metrics_formatted.txt"
logger = logging.getLogger(__name__)


def remove_cache_dirs(dir_list):
    tmp_list = list(dir_list)
    for i in range(len(tmp_list)-1, -1, -1):
        if ".pytest_cache" in tmp_list[i]:
            del tmp_list[i]
        elif "__pycache__" in tmp_list[i]:
            del tmp_list[i]
    return tmp_list

def delete_file(file_path):
    # Check if the file exists before deleting it
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"File '{file_path}' deleted successfully.")
        except Exception as e:
            logger.error(f"An error occurred while deleting the file: {e}")
    else:
        logger.info(f"File '{file_path}' does not exist. Continuing with the script.")

def get_platform(board_str):
    return "alveo" if "U250" in board_str else "zynq-iodma"

def get_full_parameterized_test_list(marker, test_dir_list, batch_size_list, platform_list):
    test_cases = [
            (f'{marker}_{param1}_batchSize-{param2}_platform-{param3}', {
            'test_dir': param1,
            'batch_size': param2,
            'platform': param3,
        })
        for param1, param2, param3 in itertools.product(
            test_dir_list,
            batch_size_list,
            platform_list,
        )
    ]
    return test_cases

def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    scenarios = []

    # Separate the full list of markers used on command line.
    # This allows a user to select multiple markers
    all_markers_used =  metafunc.config.getoption("-m").split(" ")
    current_dir = os.getcwd()
    test_dirs = [name for name in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, name))]
    test_dirs = remove_cache_dirs(test_dirs)

    for marker in all_markers_used:
        if "Pynq" in marker or "U250" in marker or "ZCU104" in marker or "KV260_SOM" in marker:
            platform = get_platform(marker)
            scenarios.extend(get_full_parameterized_test_list(marker, test_dir_list=test_dirs, batch_size_list=[1], platform_list=[platform]))

    if len(scenarios) > 0:
        for scenario in scenarios:
            idlist.append(scenario[0])
            items = scenario[1].items()
            argnames = [x[0] for x in items]
            argvalues.append([x[1] for x in items])
        metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


@pytest.mark.Pynq
@pytest.mark.U250
@pytest.mark.ZCU104
@pytest.mark.KV260_SOM
class TestBnn:
    def test_type_execute(self, test_dir, batch_size, platform):
        # Enter into test directory and clean any files from a potential previous run
        os.chdir(os.path.join(base_dir_global, test_dir))
        delete_file(output_execute_results_file)

        # Run test option: execute
        bitfile = "a.xclbin" if platform == "alveo" else "resizer.bit"
        result = subprocess.run(["python", "driver.py", "--exec_mode=execute", f"--batchsize={batch_size}", f"--bitfile={bitfile}", "--inputfile=input.npy", "--outputfile=output.npy", f"--platform={platform}"], capture_output=True, text=True, timeout=default_test_run_timeout)
        assert result.returncode == 0
        
        # Load the output and reference arrays
        output_array = np.load(output_execute_results_file)
        reference_array = np.load(execute_results_reference_file)

        # Compare the arrays
        try:
            assert np.isclose(output_array, reference_array).all()
        except AssertionError as e:
            logger.error("AssertionError occurred: %s", e, exc_info=True)
            raise

    def test_type_throughput(self, test_dir, batch_size, platform):
        os.chdir(os.path.join(base_dir_global, test_dir))
        delete_file(output_throughput_results_file)

        # Run test option: throughput
        bitfile = "a.xclbin" if platform == "alveo" else "resizer.bit"
        result = subprocess.run(["python", "driver.py", "--exec_mode=throughput_test", f"--batchsize={batch_size}", f"--bitfile={bitfile}", "--inputfile=input.npy", "--outputfile=output.npy", f"--platform={platform}"], capture_output=True, text=True, timeout=default_test_run_timeout)
        assert result.returncode == 0

        # Check if nw_metrics.txt now exists after test run
        assert os.path.exists(output_throughput_results_file)

        with open(output_throughput_results_file, "r") as file:
            res = eval(file.read())

        # try a range of batch sizes, some may fail due to insufficient DMA
        # buffers
        bsize_range_in = [8**i for i in range(5)]
        bsize_range = []
        ret = dict()
        for bsize in bsize_range_in:
            if res is not None:
                ret[bsize] = res
                bsize_range.append(bsize)
            else:
                # assume we reached largest possible N
                break

        y = [ret[key]["runtime[ms]"] for key in bsize_range]
        lrret = linregress(bsize_range, y)
        ret_str = ""
        ret_str += "\n" + "%s Throughput Test Results" % test_dir
        ret_str += "\n" + "-----------------------------"
        ret_str += "\n" + "From linear regression:"
        ret_str += "\n" + "Invocation overhead: %f ms" % lrret.intercept
        ret_str += "\n" + "Time per sample: %f ms" % lrret.slope
        ret_str += "\n" + "Raw data:"

        ret_str += "\n" + "{:<8} {:<16} {:<16} {:<16} {:<16} {:<16}".format(
            "N", "runtime[ms]", "fclk[mhz]", "fps", "DRAM rd[MB/s]", "DRAM wr[MB/s]"
        )
        for k in bsize_range:
            v = ret[k]
            ret_str += "\n" + "{:<8} {:<16} {:<16} {:<16} {:<16} {:<16}".format(
                k,
                np.round(v["runtime[ms]"], 4),
                v["fclk[mhz]"],
                np.round(v["throughput[images/s]"], 2),
                np.round(v["DRAM_in_bandwidth[MB/s]"], 2),
                np.round(v["DRAM_out_bandwidth[MB/s]"], 2),
            )
        ret_str += "\n" + "-----------------------------"
        largest_bsize = bsize_range[-1]
        
        # Dump the metrics to a text file
        with open(throughput_results_formatted_file, "w") as f:
            f.write(ret_str)
        assert os.path.exists(throughput_results_formatted_file)