# Status for FINN example networks

FINN uses [several pre-trained QNNs](https://github.com/maltanar/brevitas_cnv_lfc) that serve as examples and testcases. You can find a status summary below for each network.

* TFC, SFC, LFC... are fully-connected networks trained on the MNIST dataset
* CNV is a convolutional network trained on the CIFAR-10 dataset
* w_a_ refers to the quantization used for the weights (w) and activations (a) in bits


|                           	| Basic test | TFC-w1a1 	| TFC-w1a2 	| CNV-w1a1 	| CNV-w1a2 	| CNV-w2a2 	|
|---------------------------	|------------ |----------	|----------	|----------	|----------	|----------	|
| Export/Import             	| x           | x        	| x        	| x        	|          	|          	|
| Streamlining              	| x           | x        	| x        	|          	|          	|          	|
| Convert to HLS layers     	| x           | x        	|          	|          	|          	|          	|
| npysim                    	| x           | x        	|          	|          	|          	|          	|
| Stitched IPI design        	| x           | x        	|          	|          	|          	|           |
| rtlsim node-by-node        	| x           | x        	|          	|          	|          	|          	|
| rtlsim stitched IP        	| x           | x        	|          	|          	|          	|          	|
| Hardware test             	| x           | x        	|          	|          	|          	|          	|
