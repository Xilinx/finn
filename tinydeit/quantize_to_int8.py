#!/usr/bin/env python3
"""
Apply PTQ Quantization using Brevitas to FP32 Model and Export to Clean ONNX
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from datasets import load_dataset
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Uint8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.graph import ModuleToModuleByInstance
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.quantize import layerwise_quantize
# from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_sdpa_with_quantizable_layers
from brevitas.graph import TorchFunctionalToModule
from brevitas.nn import ScaledDotProductAttention
import torch.nn.functional as F
from transformers.utils.fx import symbolic_trace
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.general import (
    RemoveUnusedTensors,
    SortGraph,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)


def replace_sdpa_with_quantizable_layers(model):
    """Replace scaled dot product attention with quantizable version"""
    fn_to_module_map = ((F.scaled_dot_product_attention, ScaledDotProductAttention),)
    model = TorchFunctionalToModule(fn_to_module_map=fn_to_module_map).apply(model)
    return model


def create_tinybert_config():
    """Create TinyBERT configuration"""
    config = BertConfig(
        vocab_size=30522,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=1536,
        hidden_act="relu",
        num_labels=2
    )
    return config


def load_fp32_model(model_path, max_length=128):
    """Load the trained FP32 model"""
    print(f"Loading FP32 model from {model_path}...")
    config = create_tinybert_config()
    model = BertForSequenceClassificationWrapper(config, max_length)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    model.eval()
    return model


def apply_bert_quantization(model, config, bitwidth=8, seqlen=128):
    """Apply BERT-style quantization using layerwise approach"""
    print(f"Applying BERT-style quantization with {bitwidth}-bit precision...")

    dtype = torch.float32
    model.to(dtype=dtype)
    model.eval()
    vocab_size = model.config.vocab_size
    batch_size = 1

    input_ids = torch.randint(vocab_size, (batch_size, seqlen), dtype=torch.int64)
    inp = {'input_ids': input_ids}

    print("Performing symbolic tracing...")
    input_names = inp.keys()
    model = symbolic_trace(model, input_names, disable_check=True)

    print("Replacing SDPA with quantizable variants...")
    model = replace_sdpa_with_quantizable_layers(model)
    print("Replacement done.")

    unsigned_hidden_act = config.hidden_act == 'relu'
    layerwise_compute_layer_map = {}

    # Linear layer quantization
    layerwise_compute_layer_map[nn.Linear] = (
        qnn.QuantLinear,
        {
            'input_quant': lambda module: Uint8ActPerTensorFloat
                if module.in_features == config.intermediate_size and unsigned_hidden_act
                else Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'weight_bit_width': bitwidth,
            'output_quant': None,
            'bias_quant': None,
            'return_quant_tensor': False
        }
    )

    layerwise_compute_layer_map[qnn.ScaledDotProductAttention] = (
        qnn.QuantScaledDotProductAttention,
        {
            'softmax_input_quant': Int8ActPerTensorFloat,
            'softmax_input_bit_width': bitwidth,
            'attn_output_weights_quant': Uint8ActPerTensorFloat,
            'attn_output_weights_bit_width': bitwidth,
            'q_scaled_quant': Int8ActPerTensorFloat,
            'q_scaled_bit_width': bitwidth,
            'k_transposed_quant': Int8ActPerTensorFloat,
            'k_transposed_bit_width': bitwidth,
            'v_quant': Int8ActPerTensorFloat,
            'v_bit_width': bitwidth,
            'out_quant': Int8ActPerTensorFloat,
            'out_bit_width': bitwidth,
            'return_quant_tensor': False
        }
    )

    # HardTanh quantization (replacing Tanh)
    layerwise_compute_layer_map[nn.Tanh] = (
        qnn.QuantHardTanh,
        {
            'input_quant': None,
            'act_quant': Int8ActPerTensorFloat,
            'act_bit_width': bitwidth,
            'min_val': -1.0,
            'max_val': 1.0,
            'return_quant_tensor': False
        }
    )

    print("Applying layerwise quantization...")
    model = layerwise_quantize(
        model=model,
        compute_layer_map=layerwise_compute_layer_map
    )
    model.to(dtype=dtype)

    print("BERT quantization completed.")
    return model


def calibrate_model(model, tokenizer, num_samples=1600, max_length=128):
    """Calibrate the quantized model with sample data using proper calibration mode"""
    print(f"Calibrating model with ~{num_samples} samples...")

    dataset = load_dataset("glue", "sst2")
    calibration_samples = dataset["train"].shuffle(seed=42).select(range(num_samples))

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    calibration_data = calibration_samples.map(tokenize_function, batched=True)
    calibration_data.set_format(type="torch", columns=["input_ids"])
    calibration_dataloader = DataLoader(calibration_data, batch_size=32, shuffle=False)

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad(), calibration_mode(model):
        for batch_idx, batch in enumerate(tqdm(calibration_dataloader, desc="Calibrating")):
            input_ids = batch["input_ids"].to(device)

            _ = model(input_ids)

            if batch_idx >= 50:
                break

    print("Calibration completed")

class BertForSequenceClassificationWrapper(BertForSequenceClassification):
    def __init__(self, config, max_length=128):
        super().__init__(config)
        self.max_length = max_length

    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        attention_mask = torch.ones((batch_size, self.max_length), dtype=torch.long, device=input_ids.device)
        return super().forward(input_ids=input_ids, attention_mask=attention_mask)


def apply_qonnx_cleanup(model_path):
    """Apply QONNX cleanup transformations to reduce complexity"""

    try:
        model = ModelWrapper(model_path)

        print(f"  Original model has {len(model.graph.node)} nodes")

        model = model.transform(InferDataTypes())
        model = model.transform(InferShapes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveUniqueParameterTensors())
        model = model.transform(SortGraph())
        model = model.transform(FoldConstants())
        model = model.transform(RemoveUnusedTensors())

        model = model.transform(FoldTransposeIntoQuantInit())

        print(f"  Cleaned model has {len(model.graph.node)} nodes")

        cleaned_path = model_path.replace('.onnx', '_cleaned.onnx')
        model.save(cleaned_path)

        print(f"  Cleaned model saved to: {cleaned_path}")
        return cleaned_path

    except Exception as e:
        print(f"  QONNX cleanup failed: {e}")
        return model_path


def export_quantized_to_onnx(model, output_path, max_length=128):
    """Export quantized model to clean ONNX"""
    device = next(model.parameters()).device
    model.eval()

    dummy_input = torch.ones(1, max_length, dtype=torch.long).to(device)

    from brevitas.export import export_qonnx
    print(f"Attempting QONNX export with dynamo=True...")
    export_qonnx(model, dummy_input, output_path, dynamo=True)
    print(f"QONNX export successful")

    print(f"Quantized ONNX model saved to: {output_path}")
    cleaned_path = apply_qonnx_cleanup(output_path)

    return cleaned_path


def validate_quantized_model(original_model, quantized_model, tokenizer, max_length=128):
    print("Validating quantized model...")

    dataset = load_dataset("glue", "sst2")
    test_samples = dataset['validation'].shuffle(seed=42).select(range(100))

    original_model.eval()
    quantized_model.eval()
    device = next(quantized_model.parameters()).device

    original_correct = 0
    quantized_correct = 0

    with torch.no_grad():
        for sample in test_samples:
            # Tokenize
            inputs = tokenizer(
                sample['sentence'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = inputs['input_ids'].to(device)
            true_label = sample['label']

            orig_outputs = original_model(input_ids)
            orig_pred = torch.argmax(orig_outputs.logits, dim=-1).item()
            if orig_pred == true_label:
                original_correct += 1

            quant_outputs = quantized_model(input_ids)
            # Handle different output formats
            if hasattr(quant_outputs, 'logits'):
                quant_logits = quant_outputs.logits
            elif isinstance(quant_outputs, dict) and 'logits' in quant_outputs:
                quant_logits = quant_outputs['logits']
            else:
                # If it's a tensor or other format, assume it's the logits directly
                quant_logits = quant_outputs
            quant_pred = torch.argmax(quant_logits, dim=-1).item()
            if quant_pred == true_label:
                quantized_correct += 1

    orig_acc = original_correct / len(test_samples) * 100
    quant_acc = quantized_correct / len(test_samples) * 100

    print(f"Original model accuracy: {orig_acc:.2f}%")
    print(f"Quantized model accuracy: {quant_acc:.2f}%")
    print(f"Accuracy difference: {quant_acc - orig_acc:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Quantize FP32 Model to INT8 and Export to ONNX')
    parser.add_argument('--input_model', default='best_fp32_model.pth',
                        help='Path to FP32 PyTorch model')
    parser.add_argument('--output', default='quantized_int8_model.onnx',
                        help='Output quantized ONNX path')
    parser.add_argument('--calibration_samples', type=int, default=1600,
                        help='Number of samples for calibration')
    parser.add_argument('--bitwidth', type=int, default=8,
                        help='Quantization bit width')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--validate', action='store_true',
                        help='Validate quantized model accuracy')

    args = parser.parse_args()

    if not os.path.exists(args.input_model):
        print(f"Error: Input model not found at {args.input_model}")
        print("Please run train_fp32_model.py first")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    original_model = load_fp32_model(args.input_model, args.max_length)
    original_model.to(device)

    config = create_tinybert_config()
    quantized_model = apply_bert_quantization(original_model, config, args.bitwidth, args.max_length)
    quantized_model.to(device)

    print(f"Quantized model has {sum(p.numel() for p in quantized_model.parameters()):,} parameters")

    calibrate_model(quantized_model, tokenizer, args.calibration_samples, args.max_length)

    if args.validate:
        validate_quantized_model(original_model, quantized_model, tokenizer, args.max_length)

    cleaned_model_path = export_quantized_to_onnx(quantized_model, args.output, args.max_length)

    torch.save(quantized_model.state_dict(), 'quantized_int8_model.pth')

    print(f"\nQuantization completed!")
    print(f"Quantized ONNX model saved to: {args.output}")
    if cleaned_model_path != args.output:
        print(f"Cleaned ONNX model saved to: {cleaned_model_path}")
    print(f"Quantized PyTorch model saved to: quantized_int8_model.pth")


if __name__ == "__main__":
    main()
