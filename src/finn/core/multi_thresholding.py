import numpy as np


def compare(value, threshold):
        if (value >= threshold):
                res = 1.0
        else:
                res = 0.0
        return res

def execute(v,thresholds):
	#reshape inputs to enable channel-wise reading
        vr = inputs.reshape((thresholds.shape[1], -1))
        
	#calculate the channel interval for the for loops
	num_channels = thresholds.shape[0]
        channel_interval = int(inputs_reshaped.shape[1]/num_channels)
 	
	#iniate output tensor 
	ret = np.zeros_like(vr)
        
	#initiate helper variable i for channel-wise thresholding
	i = -1

	#iterate over thresholds channel-wise
        for t in thresholds:
                i += 1
		#calculate the lower and upper limit in which elements belong to one channel 
                if i == 0:
                        ce1_low_lim = 0
                else:
                        ce1_low_lim = i*channel_interval
                ce1_up_lim = (i+1)*channel_interval

		#iterate in ascending order over the thresholds belonging to one channel
                for c in range(thresholds.shape[1]):
                        for ce0 in range(inputs_reshaped.shape[0]):
                                for ce1 in range(ce1_low_lim,ce1_up_lim):
                                        ret[ce0][ce1] += compare(vr[ce0][ce1], t[c])


        return ret.reshape(inputs.shape)

