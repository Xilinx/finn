#!/usr/bin/env python3
"""
Train/Fine-tune TinyDEIT on CIFAR-10
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, classification_report
import os
from PIL import Image
import json
from tqdm import tqdm


def load_cifar10_dataset(processor, validation_split=0.1):
    train_dataset = load_dataset("cifar10", split="train")
    test_dataset = load_dataset("cifar10", split="test")
    class_names = train_dataset.features["label"].names
    
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = train_dataset.select(range(train_size, train_size + val_size))
    train_dataset = train_dataset.select(range(train_size))
    
    def preprocess_function(examples):
        images = [img.convert("RGB") for img in examples["img"]]
        processed = processor(images, return_tensors="pt")
        return {
            "pixel_values": processed["pixel_values"],
            "labels": examples["label"]
        }
    
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
    test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)
    
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")
    
    return train_dataset, val_dataset, test_dataset, class_names


def replace_activations_with_relu(model):
    try:
        from transformers.activations import GELUActivation
    except ImportError:
        GELUActivation = None
    
    replaced_count = 0
    def replace_recursive(module):
        nonlocal replaced_count
        for name, child in module.named_children():
            if ((GELUActivation and isinstance(child, GELUActivation)) or 
                isinstance(child, (nn.GELU, nn.SiLU)) or
                child.__class__.__name__ in ['GELU', 'SiLU', 'Swish']):
                setattr(module, name, nn.ReLU())
                replaced_count += 1
            else:
                replace_recursive(child)
    
    replace_recursive(model)
    return replaced_count


def create_cifar10_model(model_name="facebook/deit-tiny-patch16-224", num_classes=10, use_relu=False):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    if use_relu:
        replaced_count = replace_activations_with_relu(model)
    
    return model, processor


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "eval_loss": float(np.mean((predictions - labels) ** 2))
    }


def evaluate_model(model, test_dataset, class_names, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch in test_dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return accuracy, all_predictions, all_labels


def export_model_to_onnx(model, processor, output_dir, device):
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        with torch.no_grad():
            _ = model(dummy_input)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['pixel_values'],
            output_names=['logits'],
            dynamic_axes=None,
            do_constant_folding=True,
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=17
        )
    
    return onnx_path


def save_model_and_metrics(model, processor, accuracy, output_dir="./tinydeit_cifar10_model", device=None):
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    if device is not None:
        export_model_to_onnx(model, processor, output_dir, device)
    
    metrics = {
        "test_accuracy": float(accuracy),
        "num_classes": 10,
        "dataset": "CIFAR-10",
        "model_type": "TinyDEIT"
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Train TinyDEIT on CIFAR-10')
    parser.add_argument('--model_name', default='facebook/deit-tiny-patch16-224',
                        help='Base model name from HuggingFace')
    parser.add_argument('--output_dir', default='./tinydeit_cifar10_model',
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluation frequency')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Save frequency')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='Fraction of training data for validation')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--no_relu', action='store_true',
                        help='Keep GELU/SiLU activations (default: replace with ReLU)')
    
    args = parser.parse_args()
    
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, processor = create_cifar10_model(args.model_name, num_classes=10, use_relu=not args.no_relu)
    model.to(device)
    
    train_dataset, val_dataset, test_dataset, class_names = load_cifar10_dataset(
        processor, args.validation_split
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        report_to=None,  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    train_result = trainer.train()
    
    test_accuracy, predictions, labels = evaluate_model(
        model, test_dataset, class_names, device
    )
    
    model_dir = save_model_and_metrics(model, processor, test_accuracy, args.output_dir, device)
    
if __name__ == "__main__":
    main()
