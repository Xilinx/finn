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
    # C : Number of channels (must be the same value as C in input tensor)
    # B : Desired activation steps => i.e. for 4-bit activation, B=7 (2^(n)-1 and n=4)

    # assert if channel sizes do not match
    assert v.shape[1] == thresholds.shape[0]

    num_batch = v.shape[0]
    num_channel = v.shape[1]

    num_act = thresholds.shape[1]

    # reshape inputs to enable channel-wise reading
    vr = v.reshape((v.shape[0], v.shape[1], -1))

    num_img_elem = vr.shape[2]

    # initiate output tensor
    ret = np.zeros_like(vr)

    # iterate over thresholds channel-wise
    for t in range(num_channel):
        channel_thresh = thresholds[t]
        for b in range(num_batch):
            for elem in range(num_img_elem):
                print(vr[b][t][elem])
                for a in range(num_act):
                    print(channel_thresh[a])
                    ret[b][t][elem] += compare(vr[b][t][elem], channel_thresh[a])
    print(ret)
    return ret.reshape(v.shape)
