.PHONY: all clean

all: output_ultranet_fpga 

ultranet_brevitas_pretrained.pt: convert_pretrained_weights.py
	python convert_pretrained_weights.py

ultranet_qonnx_finn.onnx: ultranet_brevitas_pretrained.pt
	python export_qonnx.py

output_ultranet_fpga: ultranet_qonnx_finn.onnx
	python build.py

clean:
	rm -rf ultranet_brevitas_pretrained.pt
	rm -rf input.npy
	rm -rf ultranet_qonnx_finn.onnx
	rm -rf ultranet_finn_converted.onnx
	rm -rf expected_output.npy
	rm -rf __pycache__
	rm -rf output_ultranet_fpga
