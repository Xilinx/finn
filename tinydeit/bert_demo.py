############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
from onnxsim import simplify
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.util.cleanup import cleanup

import custom_steps  # Import custom steps to trigger registration

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from brainsmith import forge

warnings.simplefilter("ignore")


def generate_bert_model(args):
    """Load BERT model from specified ONNX file."""
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    model = onnx.load(args.model_path)
    return model


def run_brainsmith_dse(model, args):
    """Run Brainsmith with new execution tree architecture."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, "intermediate_models")
    os.makedirs(model_dir, exist_ok=True)

    onnx.save(model, os.path.join(args.output_dir, "input.onnx"))

    # Get blueprint path from args
    blueprint_path = Path(__file__).parent / args.blueprint
    
    # Forge the FPGA accelerator
    print("Forging FPGA accelerator...")
    results = forge(
        model_path=os.path.join(args.output_dir, "input.onnx"),
        blueprint_path=str(blueprint_path),
        output_dir=args.output_dir
    )
    
    # Results are automatically logged by forge()
    # Just check if we succeeded
    stats = results.stats
    if stats['successful'] == 0:
        raise RuntimeError(f"No successful builds")
    
    # The new execution tree handles output automatically
    final_model_dst = os.path.join(args.output_dir, "output.onnx")
    
    # Find the output from the successful execution
    for segment_id, result in results.segment_results.items():
        if result.success and result.output_model:
            shutil.copy2(result.output_model, final_model_dst)
            break
    
    # Handle shell metadata (matches old hw_compiler.py)
    handover_file = os.path.join(args.output_dir, "stitched_ip", "shell_handover.json")
    if os.path.exists(handover_file):
        with open(handover_file, "r") as fp:
            handover = json.load(fp)
        handover["num_layers"] = args.num_hidden_layers
        with open(handover_file, "w") as fp:
            json.dump(handover, fp, indent=4)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='BERT FINN demo using pre-trained ONNX model'
    )
    
    # Model configuration
    parser.add_argument('-o', '--output', help='Output build directory name', required=True)
    parser.add_argument('-m', '--model', dest='model_path', help='Path to ONNX model file', required=True)
    parser.add_argument('-z', '--hidden_size', type=int, default=384, 
                       help='BERT hidden_size parameter')
    parser.add_argument('-n', '--num_attention_heads', type=int, default=12, 
                       help='BERT num_attention_heads parameter')
    parser.add_argument('-l', '--num_hidden_layers', type=int, default=1, 
                       help='Number of hidden layers')
    parser.add_argument('-i', '--intermediate_size', type=int, default=1536, 
                       help='BERT intermediate_size parameter')
    parser.add_argument('-b', '--bitwidth', type=int, default=8, 
                       help='Quantization bitwidth (4 or 8)')
    parser.add_argument('-q', '--seqlen', type=int, default=128, 
                       help='Sequence length parameter')
    
    # Blueprint configuration
    parser.add_argument('--blueprint', type=str, default='bert_demo.yaml',
                       help='Blueprint YAML file to use (default: bert_demo.yaml)')
    
    args = parser.parse_args()
    
    # Determine output directory
    build_dir = os.environ.get("BSMITH_BUILD_DIR", "./build")
    print(build_dir)
    args.output_dir = os.path.join(build_dir, args.output)
    
    print("=" * 70)
    print("BERT Demo Using Brainsmith DSE")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Hidden layers: {args.num_hidden_layers}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Attention heads: {args.num_attention_heads}")
    print(f"  Intermediate size: {args.intermediate_size}")
    print(f"  Bitwidth: {args.bitwidth}")
    print(f"  Sequence length: {args.seqlen}")
    print(f"  Blueprint: {args.blueprint}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 70)
    
    try:
        # Step 1: Generate BERT model
        print("\nStep 1: Generating quantized BERT model...")
        model = generate_bert_model(args)
        
        # Step 2: Run Brainsmith DSE
        print("\nStep 2: Running Brainsmith DSE pipeline...")
        result = run_brainsmith_dse(model, args)
        
        print("\n" + "=" * 70)
        print("BUILD COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"\nERROR: Build failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
