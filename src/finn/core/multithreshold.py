import numpy as np


def compare(x, y):
    if x >= y:
        return 1.0
    else:
        return 0.0


def execute(v, thresholds):

    # the inputs are expected to be in the shape (N,C,H,W)
    # N : Batch size
    # C : Number of channels
    # H : Heigth of the input images
    # W : Width of the input images
    #
    # the thresholds are expected to be in the shape (C, B)
    # C : Number of channels (must be the same value as C in input tensor or 1
    #     if all channels use the same threshold value)
    # B : Desired activation steps => i.e. for 4-bit activation, B=7 (2^(n)-1 and n=4)

    # assert threshold shape
    is_global_threshold = thresholds.shape[0] == 1
    assert (v.shape[1] == thresholds.shape[0]) or is_global_threshold

    # save the required shape sizes for the loops (N, C and B)
    num_batch = v.shape[0]
    num_channel = v.shape[1]

    num_act = thresholds.shape[1]

    # reshape inputs to enable channel-wise reading
    vr = v.reshape((v.shape[0], v.shape[1], -1))

    # save the new shape size of the images
    num_img_elem = vr.shape[2]

    # initiate output tensor
    ret = np.zeros_like(vr)

    # iterate over thresholds channel-wise
    for t in range(num_channel):
        channel_thresh = thresholds[0] if is_global_threshold else thresholds[t]

        # iterate over batches
        for b in range(num_batch):

            # iterate over image elements on which the thresholds should be applied
            for elem in range(num_img_elem):

                # iterate over the different thresholds that correspond to one channel
                for a in range(num_act):
                    # apply successive thresholding to every element of one channel
                    ret[b][t][elem] += compare(vr[b][t][elem], channel_thresh[a])

    return ret.reshape(v.shape)
