import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantReLU, QuantIdentity
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8Bias


# Custom 4-bit quantizers for FINN compatibility
class Int4WeightQuant(Int8WeightPerTensorFloat):
    bit_width = 4

# FINN-compatible ReLU activation quantizer: unsigned, non-narrow
class Int4ActQuantUnsigned(Int8ActPerTensorFloat):
    bit_width = 4
    narrow_range = False
    signed = False  # FINN requires unsigned for ReLU activations

# FINN-compatible input/identity activation quantizer: signed, non-narrow  
class Int4ActQuantSigned(Int8ActPerTensorFloat):
    bit_width = 4
    narrow_range = False
    signed = True  # FINN requires signed for identity activations

class Int4BiasQuant(Int8Bias):
    bit_width = 4


class UltraNetBrevitasSimple(nn.Module):
    """Simplified UltraNet model using Brevitas quantization for FINN compatibility"""
    
    def __init__(self):
        super(UltraNetBrevitasSimple, self).__init__()
        
        # Use proper quantization configuration based on documentation
        self.layers = nn.Sequential(
            # Input quantization layer - quantize input to 4-bit signed
            QuantIdentity(act_quant=Int4ActQuantSigned, return_quant_tensor=True),
            
            # First layer - now receives quantized 4-bit input
            QuantConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False,
                       weight_quant=Int4WeightQuant,
                       input_quant=Int4ActQuantSigned,
                       return_quant_tensor=True),
            nn.BatchNorm2d(16),
            QuantReLU(act_quant=Int4ActQuantUnsigned, return_quant_tensor=True),
            nn.MaxPool2d(2, stride=2),

            # Second quantized block
            QuantConv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False,
                       weight_quant=Int4WeightQuant,
                       input_quant=Int4ActQuantSigned,
                       return_quant_tensor=True),
            nn.BatchNorm2d(32),
            QuantReLU(act_quant=Int4ActQuantUnsigned, return_quant_tensor=True),
            nn.MaxPool2d(2, stride=2),

            # Second quantized block  
            QuantConv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False,
                       weight_quant=Int4WeightQuant,
                       input_quant=Int4ActQuantSigned,
                       return_quant_tensor=True),
            nn.BatchNorm2d(64),
            QuantReLU(act_quant=Int4ActQuantUnsigned, return_quant_tensor=True),
            nn.MaxPool2d(2, stride=2),

            # Third quantized block
            QuantConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False,
                       weight_quant=Int4WeightQuant,
                       input_quant=Int4ActQuantSigned,
                       return_quant_tensor=True),
            nn.BatchNorm2d(64),
            QuantReLU(act_quant=Int4ActQuantUnsigned, return_quant_tensor=True),
            nn.MaxPool2d(2, stride=2),

            # Fourth quantized block
            QuantConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False,
                       weight_quant=Int4WeightQuant,
                       input_quant=Int4ActQuantSigned,
                       return_quant_tensor=True),
            nn.BatchNorm2d(64),
            QuantReLU(act_quant=Int4ActQuantUnsigned, return_quant_tensor=True),

            # Fifth quantized block
            QuantConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False,
                       weight_quant=Int4WeightQuant,
                       input_quant=Int4ActQuantSigned,
                       return_quant_tensor=True),
            nn.BatchNorm2d(64),
            QuantReLU(act_quant=Int4ActQuantUnsigned, return_quant_tensor=True),

            # Sixth quantized block
            QuantConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False,
                       weight_quant=Int4WeightQuant,
                       input_quant=Int4ActQuantSigned,
                       return_quant_tensor=True),
            nn.BatchNorm2d(64),
            QuantReLU(act_quant=Int4ActQuantUnsigned, return_quant_tensor=True),

            # Seventh quantized block
            QuantConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False,
                       weight_quant=Int4WeightQuant,
                       input_quant=Int4ActQuantSigned,
                       return_quant_tensor=True),
            nn.BatchNorm2d(64),
            QuantReLU(act_quant=Int4ActQuantUnsigned, return_quant_tensor=True),

            # Final layer
            QuantConv2d(64, 36, kernel_size=1, stride=1, padding=0,
                       weight_quant=Int4WeightQuant,
                       input_quant=Int4ActQuantSigned,
                       return_quant_tensor=False)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

    def get_backbone_features(self, x):
        """Extract just the backbone features (without YOLO head) for FINN export"""
        return self.forward(x)


if __name__ == "__main__":
    # Test the model
    model = UltraNetBrevitasSimple()
    model.eval()
    
    # Test input
    test_input = torch.randn(1, 3, 416, 416)
    
    # Test forward pass
    with torch.no_grad():
        output = model(test_input)
        print(f"Output shape: {output.shape}")
        print("Model test passed!")
