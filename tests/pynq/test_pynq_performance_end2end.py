import os

import pytest
import numpy as np
from scipy.stats import linregress
import warnings
from finn.util.test import load_test_checkpoint_or_skip
from finn.core.throughput_test import throughput_test

build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]


@pytest.mark.parametrize("end2end_example", ["tfc_w1a1", "cnv_w1a1"])
def test_pynq_performance_end2end(end2end_example):
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_%s_pynq_deploy.onnx" % end2end_example
    )
    try:
        ip = os.environ["PYNQ_IP"]  # NOQA
        board = os.environ["PYNQ_BOARD"]  # NOQA
        if ip == "" or board == "":
            pytest.skip("PYNQ board or IP address not specified")
        ret = dict()
        bsize_range = [1, 10, 100, 1000, 10000]
        for bsize in bsize_range:
            res = throughput_test(model, bsize)
            assert res is not None
            ret[bsize] = res

        y = [ret[key]["runtime[ms]"] for key in bsize_range]
        lrret = linregress(bsize_range, y)
        ret_str = ""
        ret_str += "\n" + "%s Throughput Test Results" % end2end_example
        ret_str += "\n" + "-----------------------------"
        ret_str += "\n" + "From linear regression:"
        ret_str += "\n" + "Invocation overhead: %f ms" % lrret.intercept
        ret_str += "\n" + "Time per sample: %f ms" % lrret.slope
        ret_str += "\n" + "Raw data:"

        ret_str += "\n" + "{:<8} {:<16} {:<16} {:<16} {:<16} {:<16}".format(
            "N", "runtime[ms]", "fclk[mhz]", "fps", "DRAM rd[Mb/s]", "DRAM wr[Mb/s]"
        )
        for k in bsize_range:
            v = ret[k]
            ret_str += "\n" + "{:<8} {:<16} {:<16} {:<16} {:<16} {:<16}".format(
                k,
                np.round(v["runtime[ms]"], 4),
                v["fclk[mhz]"],
                np.round(v["throughput[images/s]"], 2),
                np.round(v["DRAM_in_bandwidth[Mb/s]"], 2),
                np.round(v["DRAM_out_bandwidth[Mb/s]"], 2),
            )
        ret_str += "\n" + "-----------------------------"
        warnings.warn(ret_str)

    except KeyError:
        pytest.skip("PYNQ board or IP address not specified")
