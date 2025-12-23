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
    """Load and preprocess CIFAR-10 dataset"""
    print("Loading CIFAR-10 dataset...")
    
    # Load CIFAR-10
    train_dataset = load_dataset("cifar10", split="train")
    test_dataset = load_dataset("cifar10", split="test")
    
    # Get class names
    class_names = train_dataset.features["label"].names
    print(f"CIFAR-10 classes: {class_names}")
    
    # Split training data to create validation set
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = train_dataset.select(range(train_size, train_size + val_size))
    train_dataset = train_dataset.select(range(train_size))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    def preprocess_function(examples):
        # Convert images to RGB and apply processor
        images = [img.convert("RGB") for img in examples["img"]]
        processed = processor(images, return_tensors="pt")
        
        return {
            "pixel_values": processed["pixel_values"],
            "labels": examples["label"]
        }
    
    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
    test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)
    
    # Set format for PyTorch
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")
    
    return train_dataset, val_dataset, test_dataset, class_names


def create_cifar10_model(model_name="facebook/deit-tiny-patch16-224", num_classes=10):
    """Create TinyDEIT model adapted for CIFAR-10"""
    print(f"Creating TinyDEIT model for CIFAR-10 ({num_classes} classes)...")
    
    # Load the processor
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    # Load the model and modify the classifier head
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # Allow different classifier head size
    )
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Classifier head updated for {num_classes} classes")
    
    return model, processor


def compute_metrics(eval_pred):
    """Compute accuracy and other metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "eval_loss": float(np.mean((predictions - labels) ** 2))  # Simple MSE for logging
    }


def evaluate_model(model, test_dataset, class_names, device):
    """Evaluate model on test set with detailed metrics"""
    print("\nEvaluating model on CIFAR-10 test set...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    # Create test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_predictions, 
        target_names=class_names,
        digits=4
    ))
    
    return accuracy, all_predictions, all_labels


def save_model_and_metrics(model, processor, accuracy, output_dir="./tinydeit_cifar10_model"):
    """Save the trained model and metrics"""
    print(f"\nSaving model to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and processor
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    # Save metrics
    metrics = {
        "test_accuracy": float(accuracy),
        "num_classes": 10,
        "dataset": "CIFAR-10",
        "model_type": "TinyDEIT"
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model saved successfully!")
    print(f"  - Model files: {output_dir}/")
    print(f"  - Test accuracy: {accuracy:.4f}")
    
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
    
    args = parser.parse_args()
    
    # Setup device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and processor
    model, processor = create_cifar10_model(args.model_name, num_classes=10)
    model.to(device)
    
    # Load datasets
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
    
    # Train the model
    print(f"\nStarting training for {args.epochs} epochs...")
    train_result = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Training loss: {train_result.training_loss:.4f}")
    
    # Evaluate on test set
    test_accuracy, predictions, labels = evaluate_model(
        model, test_dataset, class_names, device
    )
    
    # Save model and metrics
    model_dir = save_model_and_metrics(model, processor, test_accuracy, args.output_dir)
    
if __name__ == "__main__":
    main()
