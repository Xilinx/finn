import torch
import torch.onnx
import brevitas.onnx as bo
from ultranet_brevitas_simple import UltraNetBrevitasSimple
import numpy as np
import os


def export_ultranet_qonnx(model_path=None, export_path="ultranet_qonnx.onnx", input_shape=(1, 3, 416, 416)):
    """
    Export UltraNet Brevitas model to QONNX format
    """
    model = UltraNetBrevitasSimple()
    model.eval()
    
    if model_path and os.path.exists(model_path):
        print(f"Loading pretrained weights from {model_path}")
        try:
            weights = torch.load(model_path, map_location='cpu')
            model.load_state_dict(weights, strict=False)
            print("Pretrained weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Proceeding with random initialization...")
    
    dummy_input = torch.randn(input_shape)
    
    class BackboneExport(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            return self.model.get_backbone_features(x)
    export_model = BackboneExport(model)
    export_model.eval()
    
    bo.export_qonnx(
        export_model,
        dummy_input,
        export_path,
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )
    return export_path


def validate_qonnx_export(qonnx_path, input_shape=(1, 3, 416, 416), tolerance=1e-4):
    """
    Validate the exported QONNX model by comparing outputs with original model
    """
    import onnx
    import onnxruntime as ort
    
    print(f"Validating QONNX export at {qonnx_path}")
    
    onnx_model = onnx.load(qonnx_path)
    ort_session = ort.InferenceSession(qonnx_path)
    
    original_model = UltraNetBrevitas()
    original_model.eval()
    test_input = torch.randn(input_shape)
    
    with torch.no_grad():
        original_output = original_model.get_backbone_features(test_input)
        if hasattr(original_output, 'value'):
            original_output = original_output.value
    
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]
    
    original_np = original_output.detach().numpy()
    max_diff = np.max(np.abs(original_np - onnx_output))
    
    print(f"Maximum difference between original and ONNX outputs: {max_diff}")
    
    if max_diff < tolerance:
        print("Validation passed - outputs match within tolerance")
        return True
    else:
        print("Validation failed - outputs differ significantly")
        return False


def convert_weights_from_original(original_model_path, brevitas_model_path):
    """
    Convert weights from original UltraNet to Brevitas version
    """
    
    original_weights = torch.load(original_model_path, map_location='cpu')
    brevitas_model = UltraNetBrevitas()
    brevitas_state = brevitas_model.state_dict()
    converted_weights = {}
    
    for key, value in original_weights.items():
        if 'layers.' in key:
            new_key = key
            
            if 'weight' in key and len(value.shape) == 4:  # Conv weight
                new_key = key  # Keep same key structure for now
                converted_weights[new_key] = value
            elif 'bias' in key:  # Conv bias  
                new_key = key
                converted_weights[new_key] = value
            elif 'running_mean' in key or 'running_var' in key:  # BatchNorm
                new_key = key
                converted_weights[new_key] = value
            elif 'weight' in key and len(value.shape) == 1:  # BatchNorm weight/bias
                new_key = key
                converted_weights[new_key] = value
    
    brevitas_state.update(converted_weights)
    brevitas_model.load_state_dict(brevitas_state, strict=False)
    torch.save(brevitas_model.state_dict(), brevitas_model_path)
    return brevitas_model_path


def main():
    brevitas_weights_path = "./ultranet_brevitas_pretrained.pt"
    original_weights_path = "./ultra_net/model/ultranet_4w4a.pt"
    
    model_path = brevitas_weights_path
    qonnx_path = "./ultranet_qonnx_finn.onnx"
    
    try:
        export_path = export_ultranet_qonnx(
            model_path=model_path,
            export_path=qonnx_path,
            input_shape=(1, 3, 416, 416)
        )
        
        if os.path.exists(export_path):
            file_size = os.path.getsize(export_path) / (1024*1024)  # MB
            print(f"QONNX model created successfully: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"Export failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()
