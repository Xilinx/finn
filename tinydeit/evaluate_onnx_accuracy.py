#!/usr/bin/env python3
"""
Evaluate ONNX Model Accuracy on Validation Set
"""

import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer
from datasets import load_dataset
import argparse
import os
import time
from tqdm import tqdm


def load_onnx_model(model_path):
    """Load ONNX model with appropriate runtime"""
    print(f"Loading ONNX model from {model_path}...")
    
    is_qonnx = False
    try:
        with open(model_path, 'rb') as f:
            content = f.read(50000)  # Read more content
            if (b'qonnx.custom_op' in content or 
                b'Quant(-1)' in content or 
                b'brevitas' in content or
                b'QuantLinear' in content or
                b'qonnx:Quant' in content):
                is_qonnx = True
    except Exception:
        pass
        
    if not is_qonnx:
        try:
            import onnxruntime as ort
            test_session = ort.InferenceSession(model_path)
            test_session = None  # Clean up
        except Exception as e:
            if 'qonnx.custom_op' in str(e) or 'Quant(-1)' in str(e):
                is_qonnx = True
    
    if is_qonnx:
        print("Detected QONNX model, using QONNX runtime...")
        try:
            from qonnx.core.modelwrapper import ModelWrapper
            from qonnx.transformation.infer_shapes import InferShapes
            from qonnx.transformation.infer_datatypes import InferDataTypes
            
            model = ModelWrapper(model_path)
            
            try:
                model = model.transform(InferDataTypes())
                model = model.transform(InferShapes())
            except Exception as e:
                print(f"  - Some transformations failed: {e}")
            
            return model, 'qonnx'
            
        except ImportError:
            print("QONNX not available, falling back to ONNX Runtime...")
            return None, None
    else:
        print("Using standard ONNX Runtime...")
        try:
            session = ort.InferenceSession(model_path)
            return session, 'onnx'
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            return None, None


def predict_batch(model, model_type, input_ids_batch):
    """Predict on a batch of input_ids"""
    if model_type == 'onnx':
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        result = model.run([output_name], {input_name: input_ids_batch})
        logits = result[0]
        
    elif model_type == 'qonnx':
        from qonnx.core.onnx_exec import execute_onnx
        
        batch_logits = []
        for i in range(input_ids_batch.shape[0]):
            single_input = input_ids_batch[i:i+1]  # Keep batch dimension
            input_dict = {"input_ids": single_input}
            
            try:
                output_dict = execute_onnx(model, input_dict)
                
                output_key = list(output_dict.keys())[-1]
                logits = output_dict[output_key]
                
                if len(logits.shape) == 1:
                    logits = logits.reshape(1, -1)
                
                batch_logits.append(logits)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                batch_logits.append(np.array([[0.0, 0.0]]))
        
        logits = np.vstack(batch_logits)
    
    return logits


def evaluate_model_accuracy(model, model_type, tokenizer, max_length=128, 
                          num_samples=None, batch_size=32):
    """Evaluate model accuracy on SST-2 validation set"""
    print("Loading SST-2 validation dataset...")
    dataset = load_dataset("glue", "sst2")
    val_dataset = dataset['validation']
    
    if model_type == 'qonnx' and batch_size > 8:
        batch_size = 8
        print(f"Using batch size {batch_size} for QONNX model")
    
    if num_samples:
        val_dataset = val_dataset.select(range(min(num_samples, len(val_dataset))))
        print(f"Evaluating on {len(val_dataset)} samples")
    else:
        print(f"Evaluating on full validation set ({len(val_dataset)} samples)")
    
    correct = 0
    total = 0
    
    for i in tqdm(range(0, len(val_dataset), batch_size), desc="Evaluating"):
        batch_end = min(i + batch_size, len(val_dataset))
        batch_samples = val_dataset[i:batch_end]
        
        texts = batch_samples['sentence']
        labels = batch_samples['label']
        
        inputs = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='np'
        )
        
        input_ids = inputs['input_ids'].astype(np.int64)
        
        try:
            logits = predict_batch(model, model_type, input_ids)
            predictions = np.argmax(logits, axis=-1)
            
            for pred, true_label in zip(predictions, labels):
                if pred == true_label:
                    correct += 1
                total += 1
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            continue
    
    if total == 0:
        print("No samples were successfully processed!")
        return 0.0
    
    accuracy = correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate ONNX model accuracy')
    parser.add_argument('--model', default='quantized_int8_model.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of validation samples to use (default: all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    model, model_type = load_onnx_model(args.model)
    if model is None:
        print("Failed to load model")
        return
    
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    
    print("\nStarting accuracy evaluation...")
    start_time = time.time()
    
    accuracy = evaluate_model_accuracy(
        model, model_type, tokenizer, 
        args.max_length, args.num_samples, args.batch_size
    )
    
    eval_time = time.time() - start_time
    
    print(f"\n=== Evaluation Results ===")
    print(f"Model: {args.model}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    
    model_size = os.path.getsize(args.model) / (1024 * 1024)
    print(f"Model size: {model_size:.2f} MB")


if __name__ == "__main__":
    main()
