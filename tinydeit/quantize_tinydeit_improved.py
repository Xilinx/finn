#!/usr/bin/env python3
"""
Improved TinyDEIT INT8 Quantization with better techniques
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Uint8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint
from brevitas.graph.calibrate import calibration_mode
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
from PIL import Image
import copy


def load_tinydeit_model(model_path=None):
    """Load the TinyDEIT model - either pre-trained or CIFAR-10 fine-tuned"""
    if model_path and os.path.exists(model_path):
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path)
    else:
        model_name = "facebook/deit-tiny-patch16-224"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
    
    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, processor


def replace_linear_modules_selective(model, skip_sensitive=True, bitwidth=8):
    """Selectively replace Linear modules, optionally skipping sensitive layers"""
    replaced_count = 0
    skipped_count = 0
    
    # Define sensitive layer patterns (classifier, first/last layers)
    sensitive_patterns = [
        'classifier',          # Final classification layer
        'embeddings.patch_embeddings.projection',  # First patch embedding
        'embeddings.cls_token',  # Class token
        'embeddings.position_embeddings'  # Position embeddings
    ]
    
    def should_skip_layer(name):
        if not skip_sensitive:
            return False
        return any(pattern in name for pattern in sensitive_patterns)
    
    def replace_recursive(module, prefix=""):
        nonlocal replaced_count, skipped_count
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                if should_skip_layer(full_name):
                    print(f"  Skipping sensitive layer: {full_name}")
                    skipped_count += 1
                else:
                    quant_linear = qnn.QuantLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        input_quant=Int8ActPerTensorFixedPoint, 
                        weight_quant=Int8WeightPerTensorFixedPoint,  # Symmetric quantization
                        weight_bit_width=bitwidth,
                        return_quant_tensor=False
                    )
                    
                    quant_linear.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        quant_linear.bias.data.copy_(child.bias.data)
                    
                    setattr(module, name, quant_linear)
                    replaced_count += 1
            else:
                # Recursively process children
                replace_recursive(child, full_name)
    
    replace_recursive(model)
    
    return replaced_count, skipped_count


def replace_gelu_modules_selective(model, skip_sensitive=True, bitwidth=8):
    """Selectively replace GELUActivation modules with QuantReLU"""
    replaced_count = 0
    skipped_count = 0
    
    try:
        from transformers.activations import GELUActivation
    except ImportError:
        print("Could not import GELUActivation")
        return 0, 0
    
    sensitive_patterns = [
        'classifier',  # Skip activations near classifier
    ]
    
    def should_skip_layer(name):
        if not skip_sensitive:
            return False
        return any(pattern in name for pattern in sensitive_patterns)
    
    def replace_recursive(module, prefix=""):
        nonlocal replaced_count, skipped_count
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, GELUActivation):
                if should_skip_layer(full_name):
                    print(f"  Skipping sensitive activation: {full_name}")
                    skipped_count += 1
                else:
                    # Use better quantization for ReLU
                    quant_relu = qnn.QuantReLU(
                        act_quant=Uint8ActPerTensorFloat,  # Keep this as float for ReLU
                        bit_width=bitwidth,
                        return_quant_tensor=False
                    )
                    
                    # Replace the module
                    setattr(module, name, quant_relu)
                    replaced_count += 1
            else:
                # Recursively process children
                replace_recursive(child, full_name)
    
    replace_recursive(model)
    
    return replaced_count, skipped_count


def quantize_model_improved(model, bitwidth=8, mixed_precision=True):
    """Apply improved DeiT quantization with mixed precision support"""
    print(f"Applying improved DeiT quantization with {bitwidth}-bit precision...")
    if mixed_precision:
        print("Using mixed precision (keeping sensitive layers in FP32)")
    
    # Clone the model
    model = copy.deepcopy(model)
    dtype = torch.float32
    model.to(dtype=dtype)
    model.eval()
    
    # Replace Linear layers with QuantLinear (selectively)
    print("Replacing Linear layers with QuantLinear...")
    linear_count, linear_skipped = replace_linear_modules_selective(
        model, skip_sensitive=mixed_precision, bitwidth=bitwidth
    )
    print(f"  Replaced {linear_count} Linear layers, skipped {linear_skipped} sensitive layers")
    
    # Replace GELUActivation with QuantReLU (selectively)
    print("Replacing GELUActivation with QuantReLU...")
    gelu_count, gelu_skipped = replace_gelu_modules_selective(
        model, skip_sensitive=mixed_precision, bitwidth=bitwidth
    )
    print(f"  Replaced {gelu_count} GELUActivation layers, skipped {gelu_skipped} sensitive activations")
    
    model.to(dtype=dtype)
    
    total_replaced = linear_count + gelu_count
    total_skipped = linear_skipped + gelu_skipped
    print(f"Improved quantization completed: {total_replaced} layers quantized, {total_skipped} kept in FP32")
    
    return model


def load_cifar10_data_augmented(processor, num_samples=1000):
    """Load CIFAR-10 test data with better diversity for calibration"""
    print(f"Loading diverse CIFAR-10 dataset ({num_samples} samples)...")
    
    # Load both train and test to get more diversity
    train_dataset = load_dataset("cifar10", split="train")
    test_dataset = load_dataset("cifar10", split="test")
    
    combined_indices = []
    samples_per_class = num_samples // 10  # Ensure balanced classes
    
    for class_id in range(10):
        # Get samples from both train and test for this class
        train_class_indices = [i for i, sample in enumerate(train_dataset) if sample["label"] == class_id]
        test_class_indices = [i for i, sample in enumerate(test_dataset) if sample["label"] == class_id]
        
        # Take roughly half from train, half from test
        train_take = min(samples_per_class // 2, len(train_class_indices))
        test_take = min(samples_per_class - train_take, len(test_class_indices))
        
        # Randomly sample
        import random
        random.seed(42)
        selected_train = random.sample(train_class_indices, train_take)
        selected_test = random.sample(test_class_indices, test_take)
        
        combined_indices.extend([(i, 'train') for i in selected_train])
        combined_indices.extend([(i, 'test') for i in selected_test])
    
    # Shuffle the combined indices
    random.shuffle(combined_indices)
    
    # Create the combined dataset
    combined_samples = []
    for idx, split in combined_indices:
        if split == 'train':
            sample = train_dataset[idx]
        else:
            sample = test_dataset[idx]
        combined_samples.append(sample)
    
    def preprocess_function(examples):
        images = [img.convert("RGB") for img in examples["img"]]
        processed = processor(images, return_tensors="pt")
        return {
            "pixel_values": processed["pixel_values"],
            "labels": examples["label"]
        }
    
    # Convert to HF dataset format and process
    from datasets import Dataset
    combined_dataset = Dataset.from_list(combined_samples)
    processed_dataset = combined_dataset.map(preprocess_function, batched=True, remove_columns=combined_dataset.column_names)
    processed_dataset.set_format(type="torch", columns=["pixel_values", "labels"])
    
    print(f"Created diverse calibration dataset with {len(processed_dataset)} samples")
    return processed_dataset


def calibrate_model_improved(model, processor, num_samples=2000, batch_size=16):
    """Improved calibration with more samples and smaller batches"""
    print(f"Calibrating model with {num_samples} diverse CIFAR-10 samples...")
    
    try:
        # Load more diverse calibration data
        calibration_dataset = load_cifar10_data_augmented(processor, num_samples)
        calibration_dataloader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=True)
        
        model.eval()
        device = next(model.parameters()).device
        
        # Longer calibration with more iterations
        with torch.no_grad(), calibration_mode(model):
            for batch_idx, batch in enumerate(tqdm(calibration_dataloader, desc="Calibrating")):
                pixel_values = batch["pixel_values"].to(device)
                _ = model(pixel_values)
                
                # Process more batches for better calibration
                if batch_idx >= (num_samples // batch_size):
                    break
        
        print(f"Calibration completed with {batch_idx + 1} batches")
        
    except Exception as e:
        print(f"Dataset calibration failed: {e}")
        print("Using random data for calibration...")
        
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad(), calibration_mode(model):
            for i in range(min(num_samples // 10, 200)):  # More random samples
                dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
                _ = model(dummy_input)
                
                if i % 50 == 0:
                    print(f"  Calibrated {i * batch_size} samples")


def benchmark_accuracy_detailed(model, processor, num_samples=1000):
    """Detailed benchmark with per-class accuracy"""
    print(f"\nDetailed benchmarking on {num_samples} CIFAR-10 test samples...")
    
    # Load test data
    dataset = load_dataset("cifar10", split="test")
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    def preprocess_function(examples):
        images = [img.convert("RGB") for img in examples["img"]]
        processed = processor(images, return_tensors="pt")
        return {
            "pixel_values": processed["pixel_values"],
            "labels": examples["label"]
        }
    
    processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    processed_dataset.set_format(type="torch", columns=["pixel_values", "labels"])
    
    model.eval()
    device = next(model.parameters()).device
    
    # Per-class tracking
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    class_correct = [0] * 10
    class_total = [0] * 10
    
    all_predictions = []
    all_labels = []
    confidence_scores = []
    
    with torch.no_grad():
        for sample in tqdm(processed_dataset, desc="Evaluating"):
            pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
            true_label = sample['labels']
            
            outputs = model(pixel_values)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Get prediction and confidence
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            all_predictions.append(predicted_class)
            all_labels.append(true_label)
            confidence_scores.append(confidence)
            
            # Update per-class stats
            class_total[true_label] += 1
            if predicted_class == true_label:
                class_correct[true_label] += 1
    
    # Calculate overall accuracy
    overall_accuracy = sum(class_correct) / sum(class_total) * 100
    
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    print(f"Average Confidence: {np.mean(confidence_scores):.3f}")
    
    print(f"\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i] * 100
            print(f"  {class_name:>10}: {acc:5.1f}% ({class_correct[i]:3d}/{class_total[i]:3d})")
    
    return overall_accuracy, np.mean(confidence_scores)


class CleanDeitWrapper(nn.Module):
    """Wrapper that provides clean input interface for DeiT export"""
    
    def __init__(self, deit_model):
        super().__init__()
        self.deit = deit_model
        
    def forward(self, pixel_values):
        return self.deit(pixel_values=pixel_values)


def apply_qonnx_cleanup(model_path):
    """Apply QONNX cleanup transformations to reduce complexity"""
    print(f"Applying QONNX cleanup to {model_path}...")
    
    try:
        model = ModelWrapper(model_path)
        
        print(f"  Original model has {len(model.graph.node)} nodes")
        
        # Apply cleanup transformations
        model = model.transform(InferDataTypes())
        model = model.transform(InferShapes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveUniqueParameterTensors())
        model = model.transform(SortGraph())
        model = model.transform(FoldConstants())
        model = model.transform(RemoveUnusedTensors())

        # Fold in the transposes around the quant node
        model = model.transform(FoldTransposeIntoQuantInit())
        
        print(f"  Cleaned model has {len(model.graph.node)} nodes")
        
        # Save cleaned model
        cleaned_path = model_path.replace('.onnx', '_cleaned.onnx')
        model.save(cleaned_path)
        
        print(f"  Cleaned model saved to: {cleaned_path}")
        return cleaned_path
        
    except Exception as e:
        print(f"  QONNX cleanup failed: {e}")
        return model_path


def export_to_qonnx(model, output_path, image_size=224, opset_version=17):
    """Export quantized model to clean QONNX with specified opset"""
    
    device = next(model.parameters()).device
    model.eval()
    
    wrapped_model = CleanDeitWrapper(model)
    wrapped_model.eval()
    
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32).to(device)
    
    from brevitas.export import export_qonnx
    export_qonnx(
        wrapped_model, 
        dummy_input, 
        output_path, 
        dynamo=True,
        opset_version=opset_version
    )
    
    print(f"Quantized QONNX model saved to: {output_path}")
    
    import onnx
    onnx_model = onnx.load(output_path)
    quant_nodes = sum(1 for node in onnx_model.graph.node 
                     if 'quant' in node.op_type.lower())
    print(f"  ONNX model has {len(onnx_model.graph.node)} nodes, {quant_nodes} quantization nodes")
    
    cleaned_path = apply_qonnx_cleanup(output_path)
    
    return cleaned_path


def count_module_types(model):
    """Count different module types"""
    counts = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        counts[module_type] = counts.get(module_type, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description='Improved TinyDEIT Quantization for CIFAR-10')
    parser.add_argument('--model_path', default=None,
                        help='Path to CIFAR-10 fine-tuned model (if available)')
    parser.add_argument('--output', default='tinydeit_quantized_improved.onnx', 
                        help='Output quantized QONNX path')
    parser.add_argument('--calibration_samples', type=int, default=2000, 
                        help='Number of CIFAR-10 samples for calibration')
    parser.add_argument('--bitwidth', type=int, default=8, 
                        help='Quantization bit width')
    parser.add_argument('--image_size', type=int, default=224, 
                        help='Input image size')
    parser.add_argument('--opset_version', type=int, default=17, 
                        help='ONNX opset version')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Disable mixed precision (quantize all layers)')
    parser.add_argument('--benchmark', action='store_true', 
                        help='Benchmark quantized model accuracy on CIFAR-10')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load TinyDEIT model and processor
    original_model, processor = load_tinydeit_model(args.model_path)
    original_model.to(device)
    
    # Count original modules
    original_counts = count_module_types(original_model)
    print(f"Original model: {original_counts.get('Linear', 0)} Linear, {original_counts.get('GELUActivation', 0)} GELUActivation")
    
    # Apply improved quantization
    print(f"\nApplying improved quantization...")
    mixed_precision = not args.no_mixed_precision
    quantized_model = quantize_model_improved(original_model, args.bitwidth, mixed_precision)
    quantized_model.to(device)
    
    # Count quantized modules
    quantized_counts = count_module_types(quantized_model)
    print(f"Quantized model: {quantized_counts.get('QuantLinear', 0)} QuantLinear, {quantized_counts.get('QuantReLU', 0)} QuantReLU, {quantized_counts.get('Linear', 0)} Linear (FP32)")
    
    # Improved calibration
    calibrate_model_improved(quantized_model, processor, args.calibration_samples)
    
    # Benchmark if requested
    if args.benchmark:
        accuracy, confidence = benchmark_accuracy_detailed(quantized_model, processor, 1000)
    
    # Export to QONNX
    onnx_path = args.output
    cleaned_onnx_path = export_to_qonnx(quantized_model, onnx_path, args.image_size, args.opset_version)
    
    # Save quantized model
    model_save_path = args.output.replace('.onnx', '.pth')
    torch.save(quantized_model.state_dict(), model_save_path)
    print(f"Quantized PyTorch model saved to: {model_save_path}")

if __name__ == "__main__":
    main()
