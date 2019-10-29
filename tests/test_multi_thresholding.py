import numpy as np

import finn.core.multi_thresholding as multi_thresh


def test_execute_multi_thresholding():
    inputs = np.genfromtxt(
        "../src/finn/data/multi-thresholding/input.csv", delimiter=","
    )
    inputs = inputs.reshape(7, 3, 2, 2)

    thresholds = np.genfromtxt(
        "../src/finn/data/multi-thresholding/thresholds.csv", delimiter=","
    )
    thresholds = thresholds.reshape(3, 7)

    outputs = np.genfromtxt(
        "../src/finn/data/multi-thresholding/output.csv", delimiter=","
    )
    outputs = outputs.reshape(7, 3, 2, 2)

    results = multi_thresh.execute(inputs, thresholds)

    assert np.isclose(outputs, results, atol=1e-3).all()
