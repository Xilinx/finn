#!/usr/bin/env python3
"""
Train FP32 TinyBERT Classification Model and Export to Clean ONNX
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from datasets import load_dataset
import numpy as np
import onnx
import onnxsim
import argparse
import os
from tqdm import tqdm


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


def load_and_preprocess_data(tokenizer, max_length=128):
    """Load and preprocess SST-2 dataset"""
    print("Loading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")
    
    def tokenize_data(examples):
        return tokenizer(
            examples['sentence'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
    
    # Tokenize datasets
    train_dataset = dataset['train'].map(tokenize_data, batched=True)
    val_dataset = dataset['validation'].map(tokenize_data, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'label'])
    
    return train_dataset, val_dataset


def train_model(model, train_loader, val_loader, device, epochs=3):
    """Train the model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_pbar = tqdm(train_loader, desc="Training")
        
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_fp32_model.pth')
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    return best_val_acc


def export_to_onnx(model, tokenizer, output_path, max_length=128):
    """Export model to clean ONNX format"""
    print("Exporting to ONNX...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.ones(1, max_length, dtype=torch.long).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    # Simplify ONNX model
    print("Simplifying ONNX model...")
    model_onnx = onnx.load(output_path)
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_onnx, output_path)
    
    print(f"Clean ONNX model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train FP32 TinyBERT and Export to ONNX')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--output', default='fp32_model.onnx', help='Output ONNX path')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and create model
    print("Loading tokenizer and creating model...")
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    config = create_tinybert_config()
    model = BertForSequenceClassification(config)
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load data
    train_dataset, val_dataset = load_and_preprocess_data(tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train model
    best_acc = train_model(model, train_loader, val_loader, device, args.epochs)
    
    # Load best model for export
    model.load_state_dict(torch.load('best_fp32_model.pth'))
    model.eval()
    
    # Export to ONNX
    export_to_onnx(model, tokenizer, args.output, args.max_length)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"FP32 ONNX model saved to: {args.output}")
    print(f"PyTorch model saved to: best_fp32_model.pth")


if __name__ == "__main__":
    main()
