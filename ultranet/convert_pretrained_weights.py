#!/usr/bin/env python3
"""
Convert pretrained UltraNet weights to Brevitas model format

This script loads the original pretrained weights and maps them to the Brevitas quantized model.
"""

import torch
import numpy as np
from ultranet_brevitas_simple import UltraNetBrevitasSimple


def convert_weights_to_brevitas(checkpoint_path, output_path):
    """
    Convert original UltraNet checkpoint to Brevitas model weights
    
    Args:
        checkpoint_path: Path to original checkpoint (ultranet_4w4a.pt)
        output_path: Output path for converted Brevitas weights
    """
    
    print(f"Loading original checkpoint from {checkpoint_path}")
    
    # Load original checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    original_weights = checkpoint['model']
    
    print(f"Original model has {len(original_weights)} parameters")
    
    # Create Brevitas model
    brevitas_model = UltraNetBrevitasSimple()
    brevitas_state = brevitas_model.state_dict()
    
    print(f"Brevitas model has {len(brevitas_state)} parameters")
    
    # Weight mapping
    converted_weights = {}
    
    # Map the weights from original model to Brevitas model
    # The original model uses layers.X.weight/bias format
    # The Brevitas model has a similar structure but may have additional quantization parameters
    for orig_key, orig_value in original_weights.items():
        if orig_key in brevitas_state:
            print(f"Mapping {orig_key}: {orig_value.shape} -> {brevitas_state[orig_key].shape}")
            
            if orig_value.shape == brevitas_state[orig_key].shape:
                converted_weights[orig_key] = orig_value
            else:
                print(f"Shape mismatch for {orig_key}: {orig_value.shape} vs {brevitas_state[orig_key].shape}")
        else:
            print(f"Key {orig_key} not found in Brevitas model, skipping...")
    
    brevitas_state.update(converted_weights)
    brevitas_model.load_state_dict(brevitas_state, strict=False)
    torch.save(brevitas_model.state_dict(), output_path)
    
    print(f"Successfully mapped {len(converted_weights)} parameters")
    
    return output_path


def test_converted_model(weights_path):
    """Test the converted Brevitas model"""
    model = UltraNetBrevitasSimple()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)
    model.eval()
    
    test_input = torch.randn(1, 3, 416, 416)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"Test successful! Output shape: {output.shape}")
    return True


def main():
    """Main function"""
    original_checkpoint = "./ultra_net/model/ultranet_4w4a.pt"
    converted_weights = "./ultranet_brevitas_pretrained.pt"
    
    convert_weights_to_brevitas(original_checkpoint, converted_weights)
    test_converted_model(converted_weights)
    
    print("Weight conversion completed successfully!")
    print(f"You can now use the converted weights: {converted_weights}")


if __name__ == "__main__":
    main()
