import numpy as np


def compare(value, threshold):
        if (value >= threshold):
                res = 1.0
        else:
                res = 0.0
        return res

def execute(inputs,thresholds):

        inputs_reshaped = inputs.reshape((thresholds.shape[1], -1))
        num_channels = thresholds.shape[0]
        ret = np.zeros_like(inputs_reshaped)
        channel_interval = int(inputs_reshaped.shape[1]/num_channels)

        i=-1
        for t in thresholds:
                i+=1
                if i == 0:
                        ce1_low_lim=0
                else:
                        ce1_low_lim=i*channel_interval
                ce1_up_lim=(i+1)*channel_interval

                for c in range(thresholds.shape[1]):
                        for ce0 in range(inputs_reshaped.shape[0]):
                                for ce1 in range(ce1_low_lim,ce1_up_lim):
                                        ret[ce0][ce1] += compare(inputs_reshaped[ce0][ce1], t[c])


        return ret.reshape(inputs.shape)

